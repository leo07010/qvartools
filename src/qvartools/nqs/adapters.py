"""
adapters --- NQS adapter layers for cross-pipeline interoperability
====================================================================

Provides adapter wrappers that bridge the interface gap between the
NF training pipeline (which expects ``log_amplitude(x)``) and the
HI training pipeline (which expects ``sample()`` + ``log_prob()``).

Classes
-------
TransformerAsNQS
    Wraps :class:`AutoregressiveTransformer` to expose ``log_amplitude``
    and ``phase`` for use in the NF training pipeline.
NQSWithSampling
    Wraps any :class:`NeuralQuantumState` to expose ``sample()`` and
    ``log_prob()`` for use in the HI training pipeline.
"""

from __future__ import annotations

import logging
from itertools import combinations

import torch
import torch.nn as nn

from qvartools.nqs.neural_state import NeuralQuantumState

__all__ = [
    "TransformerAsNQS",
    "NQSWithSampling",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TransformerAsNQS: Transformer -> NF pipeline
# ---------------------------------------------------------------------------


class TransformerAsNQS(NeuralQuantumState):
    """Adapt an :class:`AutoregressiveTransformer` to the NQS interface.

    Converts the transformer's ``log_prob(alpha, beta)`` output to
    ``log_amplitude(x)`` via the relationship ``|psi(x)|^2 ~ p(x)``,
    giving ``log_amplitude(x) = 0.5 * log_prob(alpha, beta)``.

    This is an approximation: the autoregressive probability is a
    *model* of ``|psi|^2``, not a true wavefunction amplitude.
    The phase is identically zero (real-valued NQS).

    Parameters
    ----------
    transformer : AutoregressiveTransformer
        The transformer model to wrap.  Must have ``n_orbitals``,
        ``n_alpha``, ``n_beta``, ``log_prob``, and ``sample`` attributes.
    """

    def __init__(self, transformer: nn.Module) -> None:
        n_orb: int = transformer.n_orbitals  # type: ignore[attr-defined,assignment]
        num_sites = 2 * n_orb
        super().__init__(num_sites=num_sites, local_dim=2, complex_output=False)

        self._transformer = transformer
        self._n_orb = n_orb

        logger.info(
            "TransformerAsNQS adapter created: num_sites=%d, n_orb=%d",
            num_sites,
            n_orb,
        )

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-amplitude via ``0.5 * transformer.log_prob``.

        Parameters
        ----------
        x : torch.Tensor
            Configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Log-amplitudes, shape ``(batch,)``.
        """
        alpha = x[:, : self._n_orb]
        beta = x[:, self._n_orb :]
        log_p = self._transformer.log_prob(alpha, beta)  # type: ignore[attr-defined,operator]
        return 0.5 * log_p

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Return zero phase (real-valued NQS).

        Parameters
        ----------
        x : torch.Tensor
            Configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Zeros, shape ``(batch,)``.
        """
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# NQSWithSampling: NeuralQuantumState -> HI pipeline
# ---------------------------------------------------------------------------


class NQSWithSampling(nn.Module):
    """Adapt any :class:`NeuralQuantumState` for the HI training pipeline.

    Provides ``sample()`` and ``log_prob()`` by generating
    particle-conserving configurations and evaluating the wrapped NQS.

    Sampling generates all valid particle-conserving configurations
    and draws from the NQS Born-rule distribution.  For larger systems
    this should be replaced with MCMC or flow-based sampling.

    Parameters
    ----------
    nqs : NeuralQuantumState
        The NQS model to wrap.
    n_orbitals : int
        Number of spatial orbitals per spin channel.
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.
    """

    def __init__(
        self,
        nqs: NeuralQuantumState,
        n_orbitals: int,
        n_alpha: int,
        n_beta: int,
    ) -> None:
        super().__init__()
        self._nqs = nqs
        self.n_orbitals = n_orbitals
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.num_sites = 2 * n_orbitals

        # Pre-generate all valid configurations for small systems
        self._valid_configs: torch.Tensor | None = None

        logger.info(
            "NQSWithSampling adapter created: n_orb=%d, n_alpha=%d, n_beta=%d",
            n_orbitals,
            n_alpha,
            n_beta,
        )

    _MAX_ENUM_CONFIGS = 50_000

    def _get_valid_configs(self, device: torch.device) -> torch.Tensor:
        """Generate or retrieve all particle-conserving configurations."""
        if self._valid_configs is not None:
            return self._valid_configs.to(device)

        import math

        n = self.n_orbitals
        n_configs = math.comb(n, self.n_alpha) * math.comb(n, self.n_beta)
        if n_configs > self._MAX_ENUM_CONFIGS:
            raise RuntimeError(
                f"NQSWithSampling: enumeration would produce {n_configs:,} "
                f"configurations (max {self._MAX_ENUM_CONFIGS:,}). "
                f"Use MCMC or flow-based sampling for large systems."
            )
        alpha_list: list[torch.Tensor] = []
        for occ in combinations(range(n), self.n_alpha):
            row = torch.zeros(n, dtype=torch.float32)
            for i in occ:
                row[i] = 1.0
            alpha_list.append(row)
        alpha_configs = torch.stack(alpha_list)

        beta_list: list[torch.Tensor] = []
        for occ in combinations(range(n), self.n_beta):
            row = torch.zeros(n, dtype=torch.float32)
            for i in occ:
                row[i] = 1.0
            beta_list.append(row)
        beta_configs = torch.stack(beta_list)

        n_a = alpha_configs.shape[0]
        n_b = beta_configs.shape[0]
        all_configs = torch.zeros(n_a * n_b, self.num_sites, dtype=torch.float32)
        idx = 0
        for i in range(n_a):
            for j in range(n_b):
                all_configs[idx, :n] = alpha_configs[i]
                all_configs[idx, n:] = beta_configs[j]
                idx += 1

        self._valid_configs = all_configs
        return all_configs.to(device)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample particle-conserving configurations from the NQS.

        Uses Born-rule sampling from the NQS probability distribution
        over all valid particle-conserving configurations.

        Parameters
        ----------
        n_samples : int
            Number of configurations to generate.
        temperature : float, optional
            Sampling temperature (default ``1.0``).

        Returns
        -------
        torch.Tensor
            Sampled configurations, shape ``(n_samples, num_sites)``.
        """
        param = next(self._nqs.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        valid = self._get_valid_configs(device)

        log_amp = self._nqs.log_amplitude(valid)
        logits = 2.0 * log_amp / temperature
        probs = torch.softmax(logits, dim=0)

        indices = torch.multinomial(probs, n_samples, replacement=True)
        return valid[indices].long()

    def log_prob(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-probability for the (alpha, beta) configuration.

        Returns ``2 * log_amplitude(x)`` (unnormalized log-probability).

        Parameters
        ----------
        alpha : torch.Tensor
            Alpha occupations, shape ``(batch, n_orbitals)``.
        beta : torch.Tensor
            Beta occupations, shape ``(batch, n_orbitals)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities, shape ``(batch,)``.
        """
        x = torch.cat([alpha, beta], dim=-1).float()
        return 2.0 * self._nqs.log_amplitude(x)
