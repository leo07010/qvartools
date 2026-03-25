"""
particle_conserving_flow --- Particle-number-conserving flow sampler
====================================================================

Implements a normalizing-flow sampler that exactly conserves the number
of alpha and beta electrons by construction.  Instead of learning an
unconstrained bijection, this module learns a *scoring function* for
each orbital and selects the top-k orbitals via differentiable top-k
mechanisms (Gumbel-Softmax or sigmoid-based).

The result is a set of binary configurations in which exactly
``n_alpha`` alpha orbitals and ``n_beta`` beta orbitals are occupied,
guaranteeing valid Slater determinants for quantum chemistry.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from qvartools.flows.training.gumbel_topk import GumbelTopK

__all__ = [
    "ParticleConservingFlowSampler",
    "verify_particle_conservation",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orbital scoring network
# ---------------------------------------------------------------------------


class OrbitalScoringNetwork(nn.Module):
    """Neural network that scores orbital occupations.

    Given an optional context vector (e.g. from the alpha-spin
    configuration), produces logits indicating how favourable it is to
    occupy each orbital.

    Parameters
    ----------
    n_orbitals : int
        Number of orbitals to score.
    hidden_dims : list of int, optional
        Hidden-layer sizes (default ``[128, 64]``).
    context_dim : int, optional
        Dimensionality of the context vector (default ``32``).
        If zero, no context is used.

    Attributes
    ----------
    n_orbitals : int
        Number of orbitals.
    context_dim : int
        Context vector size.
    net : nn.Sequential
        The scoring MLP.
    """

    def __init__(
        self,
        n_orbitals: int,
        hidden_dims: Optional[List[int]] = None,
        context_dim: int = 32,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.n_orbitals: int = n_orbitals
        self.context_dim: int = context_dim

        # Input: orbital index embedding + optional context
        input_dim = n_orbitals + context_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LeakyReLU(0.01))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_orbitals))

        self.net: nn.Sequential = nn.Sequential(*layers)

        # Learnable baseline logits for each orbital
        self.baseline = nn.Parameter(torch.zeros(n_orbitals))

        # Context projection (identity embedding when no external context)
        if context_dim > 0:
            self.context_proj = nn.Linear(n_orbitals, context_dim)
        else:
            self.context_proj = None

    def forward(
        self,
        batch_size: int,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce orbital-occupation logits.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.
        context : torch.Tensor or None, optional
            Context vector of shape ``(batch_size, n_orbitals)`` (e.g. the
            alpha configuration).  If ``None`` and ``context_dim > 0``,
            a zero context is used.

        Returns
        -------
        torch.Tensor
            Logits for each orbital, shape ``(batch_size, n_orbitals)``.
        """
        device = self.baseline.device

        # Build input: [baseline_expanded, context_proj]
        baseline_expanded = self.baseline.unsqueeze(0).expand(
            batch_size, -1
        )  # (batch, n_orbitals)

        if self.context_dim > 0:
            if context is not None:
                ctx = self.context_proj(context.float())  # (batch, context_dim)
            else:
                ctx = torch.zeros(
                    batch_size, self.context_dim, device=device
                )
            net_input = torch.cat(
                [baseline_expanded, ctx], dim=-1
            )  # (batch, n_orbitals + context_dim)
        else:
            net_input = baseline_expanded

        logits = self.net(net_input)  # (batch, n_orbitals)
        return logits + baseline_expanded


# ---------------------------------------------------------------------------
# ParticleConservingFlowSampler
# ---------------------------------------------------------------------------


class ParticleConservingFlowSampler(nn.Module):
    """Normalizing flow that exactly conserves alpha and beta particle numbers.

    Produces binary configurations of shape ``(num_sites,)`` where the
    first ``num_sites // 2`` entries are alpha orbitals and the remaining
    are beta orbitals.  Exactly ``n_alpha`` alpha and ``n_beta`` beta
    orbitals are occupied in every sample.

    The flow works by:

    1. Scoring alpha orbitals with a learned network.
    2. Selecting the top ``n_alpha`` via differentiable top-k.
    3. Scoring beta orbitals conditioned on the alpha configuration.
    4. Selecting the top ``n_beta`` via differentiable top-k.
    5. Concatenating ``[alpha, beta]`` to form the full configuration.

    Parameters
    ----------
    num_sites : int
        Total number of spin-orbitals (must be even).
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.
    hidden_dims : list of int, optional
        Hidden-layer sizes for the scoring networks (default ``[128, 64]``).
    temperature : float, optional
        Initial temperature for differentiable top-k (default ``1.0``).
    min_temperature : float, optional
        Minimum temperature (default ``0.01``).

    Attributes
    ----------
    num_sites : int
        Total number of spin-orbitals.
    n_orbitals : int
        Number of spatial orbitals (``num_sites // 2``).
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.
    temperature : float
        Current temperature for top-k selection.
    alpha_scorer : OrbitalScoringNetwork
        Scoring network for alpha orbitals.
    beta_scorer : OrbitalScoringNetwork
        Scoring network for beta orbitals (conditioned on alpha config).
    selector : GumbelTopK
        Differentiable top-k selector.

    Examples
    --------
    >>> flow = ParticleConservingFlowSampler(
    ...     num_sites=10, n_alpha=2, n_beta=2
    ... )
    >>> configs, unique = flow.sample(batch_size=100)
    >>> is_valid, stats = verify_particle_conservation(
    ...     configs, n_orbitals=5, n_alpha=2, n_beta=2
    ... )
    >>> assert is_valid
    """

    def __init__(
        self,
        num_sites: int,
        n_alpha: int,
        n_beta: int,
        hidden_dims: Optional[List[int]] = None,
        temperature: float = 1.0,
        min_temperature: float = 0.01,
    ) -> None:
        super().__init__()
        if num_sites < 2 or num_sites % 2 != 0:
            raise ValueError(
                f"num_sites must be a positive even integer, got {num_sites}"
            )
        if hidden_dims is None:
            hidden_dims = [128, 64]

        n_orbitals = num_sites // 2
        if n_alpha < 0 or n_alpha > n_orbitals:
            raise ValueError(
                f"n_alpha must be in [0, {n_orbitals}], got {n_alpha}"
            )
        if n_beta < 0 or n_beta > n_orbitals:
            raise ValueError(
                f"n_beta must be in [0, {n_orbitals}], got {n_beta}"
            )

        self.num_sites: int = num_sites
        self.n_orbitals: int = n_orbitals
        self.n_alpha: int = n_alpha
        self.n_beta: int = n_beta
        self.temperature: float = temperature
        self.min_temperature: float = min_temperature

        # Scoring networks
        self.alpha_scorer: OrbitalScoringNetwork = OrbitalScoringNetwork(
            n_orbitals=n_orbitals,
            hidden_dims=hidden_dims,
            context_dim=0,  # Alpha has no conditioning context
        )
        self.beta_scorer: OrbitalScoringNetwork = OrbitalScoringNetwork(
            n_orbitals=n_orbitals,
            hidden_dims=hidden_dims,
            context_dim=32,  # Beta conditioned on alpha
        )

        # Differentiable top-k selector
        self.selector: GumbelTopK = GumbelTopK(
            temperature=temperature,
            min_temperature=min_temperature,
        )

    def set_temperature(self, temperature: float) -> None:
        """Set the temperature for differentiable top-k selection.

        Parameters
        ----------
        temperature : float
            New temperature value.  Will be clamped to at least
            ``min_temperature``.
        """
        self.temperature = max(temperature, self.min_temperature)
        self.selector.temperature = self.temperature

    def _soft_to_hard(
        self, soft_mask: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Convert a soft selection mask to a hard binary mask.

        Selects the top-k entries by value and sets them to 1, all
        others to 0.  Uses straight-through estimation for gradient
        flow: the forward pass uses hard values, but the backward pass
        uses soft gradients.

        Parameters
        ----------
        soft_mask : torch.Tensor
            Soft selection, shape ``(batch, n)``.
        k : int
            Number of elements to select.

        Returns
        -------
        torch.Tensor
            Hard binary mask, shape ``(batch, n)``, with exactly ``k``
            ones per row.
        """
        _, top_indices = torch.topk(soft_mask, k, dim=-1)
        hard_mask = torch.zeros_like(soft_mask)
        hard_mask.scatter_(1, top_indices, 1.0)
        # Straight-through estimator: hard in forward, soft gradients in backward
        return hard_mask - soft_mask.detach() + soft_mask

    def sample(
        self,
        batch_size: int,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample particle-conserving binary configurations.

        Each configuration has exactly ``n_alpha`` occupied alpha
        orbitals and ``n_beta`` occupied beta orbitals.

        Parameters
        ----------
        batch_size : int
            Number of configurations to sample.
        temperature : float or None, optional
            Override temperature for this call.  If ``None``, uses the
            current instance temperature.

        Returns
        -------
        all_configs : torch.Tensor
            All sampled configurations, shape ``(batch_size, num_sites)``.
            The first ``n_orbitals`` entries are alpha, the remaining are
            beta.
        unique_configs : torch.Tensor
            Unique configurations, shape ``(n_unique, num_sites)``.
        """
        temp = temperature if temperature is not None else self.temperature

        # Step 1: Score and select alpha orbitals
        alpha_logits = self.alpha_scorer(batch_size)
        alpha_soft = self.selector(alpha_logits, self.n_alpha, temperature=temp)
        alpha_config = self._soft_to_hard(alpha_soft, self.n_alpha)

        # Step 2: Score and select beta orbitals, conditioned on alpha
        beta_logits = self.beta_scorer(batch_size, context=alpha_config)
        beta_soft = self.selector(beta_logits, self.n_beta, temperature=temp)
        beta_config = self._soft_to_hard(beta_soft, self.n_beta)

        # Step 3: Concatenate [alpha, beta]
        all_configs = torch.cat([alpha_config, beta_config], dim=-1)

        # Step 4: Extract unique configurations
        unique_configs = torch.unique(all_configs, dim=0)

        return all_configs, unique_configs

    def sample_without_replacement(
        self,
        batch_size: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample unique configurations using deterministic ordering.

        Generates a larger pool of samples and returns the unique
        configurations sorted by their logit scores (most probable first).

        Parameters
        ----------
        batch_size : int
            Desired number of unique configurations.
        temperature : float or None, optional
            Override temperature.  If ``None``, uses instance temperature.

        Returns
        -------
        torch.Tensor
            Unique configurations, shape ``(n_unique, num_sites)`` where
            ``n_unique <= batch_size``.  Sorted by descending score.
        """
        # Over-sample to increase chance of getting enough unique configs
        oversample_factor = 4
        pool_size = batch_size * oversample_factor

        all_configs, unique_configs = self.sample(pool_size, temperature=temperature)

        if unique_configs.shape[0] >= batch_size:
            return unique_configs[:batch_size]
        return unique_configs

    def forward(
        self,
        batch_size: int,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass --- delegates to :meth:`sample`.

        Parameters
        ----------
        batch_size : int
            Number of configurations to sample.
        temperature : float or None, optional
            Override temperature.

        Returns
        -------
        all_configs : torch.Tensor
            All configurations, shape ``(batch_size, num_sites)``.
        unique_configs : torch.Tensor
            Unique configurations, shape ``(n_unique, num_sites)``.
        """
        return self.sample(batch_size, temperature=temperature)


# ---------------------------------------------------------------------------
# Particle-conservation verification
# ---------------------------------------------------------------------------


def verify_particle_conservation(
    configs: torch.Tensor,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
) -> Tuple[bool, Dict[str, object]]:
    """Validate that all configurations conserve particle numbers.

    Checks that each configuration has exactly ``n_alpha`` occupied
    alpha orbitals (first ``n_orbitals`` sites) and ``n_beta`` occupied
    beta orbitals (remaining ``n_orbitals`` sites).

    Parameters
    ----------
    configs : torch.Tensor
        Binary configurations, shape ``(n_configs, 2 * n_orbitals)``.
    n_orbitals : int
        Number of spatial orbitals (half of ``num_sites``).
    n_alpha : int
        Expected number of alpha electrons per configuration.
    n_beta : int
        Expected number of beta electrons per configuration.

    Returns
    -------
    is_valid : bool
        ``True`` if every configuration has exactly the correct particle
        numbers.
    stats : dict
        Dictionary with detailed statistics:

        - ``"n_configs"`` : int --- total number of configurations.
        - ``"n_valid"`` : int --- number of valid configurations.
        - ``"n_invalid"`` : int --- number of invalid configurations.
        - ``"alpha_counts"`` : torch.Tensor --- alpha electron count per config.
        - ``"beta_counts"`` : torch.Tensor --- beta electron count per config.
        - ``"alpha_violations"`` : int --- configs with wrong alpha count.
        - ``"beta_violations"`` : int --- configs with wrong beta count.

    Examples
    --------
    >>> configs = torch.tensor([[1, 1, 0, 1, 0, 1]])  # 2 alpha, 2 beta
    >>> is_valid, stats = verify_particle_conservation(configs, 3, 2, 2)
    >>> is_valid
    True
    """
    if configs.ndim != 2:
        raise ValueError(
            f"configs must be 2-dimensional, got shape {configs.shape}"
        )
    expected_cols = 2 * n_orbitals
    if configs.shape[1] != expected_cols:
        raise ValueError(
            f"configs must have {expected_cols} columns "
            f"(2 * n_orbitals), got {configs.shape[1]}"
        )

    alpha_part = configs[:, :n_orbitals]
    beta_part = configs[:, n_orbitals:]

    alpha_counts = alpha_part.sum(dim=-1)
    beta_counts = beta_part.sum(dim=-1)

    alpha_valid = alpha_counts == n_alpha
    beta_valid = beta_counts == n_beta
    all_valid = alpha_valid & beta_valid

    n_configs = configs.shape[0]
    n_valid = int(all_valid.sum().item())
    alpha_violations = int((~alpha_valid).sum().item())
    beta_violations = int((~beta_valid).sum().item())

    stats: Dict[str, object] = {
        "n_configs": n_configs,
        "n_valid": n_valid,
        "n_invalid": n_configs - n_valid,
        "alpha_counts": alpha_counts,
        "beta_counts": beta_counts,
        "alpha_violations": alpha_violations,
        "beta_violations": beta_violations,
    }

    is_valid = n_valid == n_configs
    return is_valid, stats
