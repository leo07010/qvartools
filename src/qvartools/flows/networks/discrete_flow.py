"""
discrete_flow --- RealNVP normalizing flow for discrete configurations
======================================================================

Implements a RealNVP-style normalizing flow that maps samples from a
multi-modal Gaussian prior through a sequence of affine coupling layers
to produce continuous outputs, which are then discretised into binary
configurations via thresholding.

The multi-modal prior (mixture of Gaussians centred at +/- 1) ensures
uniform coverage of both ``{0, 1}`` values after discretisation.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from qvartools.flows.networks.coupling_network import CouplingNetwork, MultiModalPrior

__all__ = [
    "DiscreteFlowSampler",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DiscreteFlowSampler
# ---------------------------------------------------------------------------


class DiscreteFlowSampler(nn.Module):
    """RealNVP normalizing flow mapping continuous latent to discrete configs.

    Uses alternating binary masks to split dimensions across coupling
    layers.  The multi-modal prior ensures uniform coverage of both
    ``{0, 1}`` values.  Continuous outputs are clamped to ``[-1, 1]``
    and discretised by thresholding at zero.

    Parameters
    ----------
    num_sites : int
        Number of binary sites in each configuration.
    num_coupling_layers : int, optional
        Number of affine coupling layers (default ``6``).
    hidden_dims : list of int, optional
        Hidden-layer sizes for the coupling networks (default ``[128, 128]``).
    prior_std : float, optional
        Standard deviation of the mixture-of-Gaussians prior components
        (default ``1.0``).
    n_mc_samples : int, optional
        Number of Monte Carlo samples for discrete log-probability
        estimation (default ``100``).

    Attributes
    ----------
    num_sites : int
        Number of sites.
    num_coupling_layers : int
        Number of coupling layers.
    prior : MultiModalPrior
        The mixture-of-Gaussians prior.
    masks : list of torch.Tensor
        Binary masks for each coupling layer.
    coupling_nets : nn.ModuleList
        Coupling networks for each layer.
    n_mc_samples : int
        Number of MC samples for discrete probability estimation.

    Examples
    --------
    >>> flow = DiscreteFlowSampler(num_sites=10, num_coupling_layers=4)
    >>> configs, unique = flow.sample(batch_size=256)
    >>> configs.shape
    torch.Size([256, 10])
    """

    def __init__(
        self,
        num_sites: int,
        num_coupling_layers: int = 6,
        hidden_dims: Optional[List[int]] = None,
        prior_std: float = 1.0,
        n_mc_samples: int = 100,
    ) -> None:
        super().__init__()
        if num_sites < 1:
            raise ValueError(f"num_sites must be >= 1, got {num_sites}")
        if num_coupling_layers < 1:
            raise ValueError(
                f"num_coupling_layers must be >= 1, got {num_coupling_layers}"
            )
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.num_sites: int = num_sites
        self.num_coupling_layers: int = num_coupling_layers
        self.n_mc_samples: int = n_mc_samples

        self.prior: MultiModalPrior = MultiModalPrior(
            num_sites=num_sites, std=prior_std
        )

        # Build alternating masks: even layers mask first half,
        # odd layers mask second half.
        self.masks: list[torch.Tensor] = []
        coupling_nets: list[CouplingNetwork] = []

        for layer_idx in range(num_coupling_layers):
            mask = torch.zeros(num_sites)
            if layer_idx % 2 == 0:
                mask[: num_sites // 2] = 1.0
            else:
                mask[num_sites // 2 :] = 1.0
            self.masks.append(mask)

            masked_dim = int(mask.sum().item())
            unmasked_dim = num_sites - masked_dim
            coupling_nets.append(
                CouplingNetwork(
                    input_dim=masked_dim,
                    hidden_dims=hidden_dims,
                    output_dim=unmasked_dim,
                )
            )

        self.coupling_nets: nn.ModuleList = nn.ModuleList(coupling_nets)

        # Register masks as buffers so they move with the model
        for i, mask in enumerate(self.masks):
            self.register_buffer(f"mask_{i}", mask)

    def _get_mask(self, layer_idx: int) -> torch.Tensor:
        """Retrieve the mask buffer for a given layer.

        Parameters
        ----------
        layer_idx : int
            Index of the coupling layer.

        Returns
        -------
        torch.Tensor
            Binary mask of shape ``(num_sites,)``.
        """
        return getattr(self, f"mask_{layer_idx}")

    def _forward_flow(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the forward (generative) direction of the flow.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples from the prior, shape ``(batch, num_sites)``.

        Returns
        -------
        y : torch.Tensor
            Transformed samples, shape ``(batch, num_sites)``.
        log_det_jacobian : torch.Tensor
            Sum of log absolute determinant of the Jacobian for each
            sample, shape ``(batch,)``.
        """
        y = z
        log_det = torch.zeros(z.shape[0], device=z.device)

        for layer_idx in range(self.num_coupling_layers):
            mask = self._get_mask(layer_idx)
            mask_b = mask.bool()

            # Split into masked (fixed) and unmasked (transformed) parts
            masked_x = y[:, mask_b]
            unmasked_x = y[:, ~mask_b]

            scale, shift = self.coupling_nets[layer_idx](masked_x)

            # Affine transform: y_unmasked = unmasked * exp(scale) + shift
            transformed = unmasked_x * torch.exp(scale) + shift
            log_det = log_det + scale.sum(dim=-1)

            # Reassemble
            y_new = torch.empty_like(y)
            y_new[:, mask_b] = masked_x
            y_new[:, ~mask_b] = transformed
            y = y_new

        return y, log_det

    def _inverse_flow(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the inverse (inference) direction of the flow.

        Parameters
        ----------
        y : torch.Tensor
            Data-space samples, shape ``(batch, num_sites)``.

        Returns
        -------
        z : torch.Tensor
            Latent-space samples, shape ``(batch, num_sites)``.
        log_det_jacobian : torch.Tensor
            Sum of log |det J^{-1}| for each sample, shape ``(batch,)``.
        """
        z = y
        log_det = torch.zeros(y.shape[0], device=y.device)

        for layer_idx in range(self.num_coupling_layers - 1, -1, -1):
            mask = self._get_mask(layer_idx)
            mask_b = mask.bool()

            masked_x = z[:, mask_b]
            unmasked_x = z[:, ~mask_b]

            scale, shift = self.coupling_nets[layer_idx](masked_x)

            # Inverse affine: z_unmasked = (unmasked - shift) * exp(-scale)
            inv_transformed = (unmasked_x - shift) * torch.exp(-scale)
            log_det = log_det - scale.sum(dim=-1)

            z_new = torch.empty_like(z)
            z_new[:, mask_b] = masked_x
            z_new[:, ~mask_b] = inv_transformed
            z = z_new

        return z, log_det

    def sample_continuous(self, batch_size: int) -> torch.Tensor:
        """Sample continuous outputs from the flow, clamped to [-1, 1].

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.

        Returns
        -------
        torch.Tensor
            Continuous samples clamped to ``[-1, 1]``, shape
            ``(batch_size, num_sites)``.
        """
        device = next(self.parameters()).device
        self.prior.device = device
        z = self.prior.sample(batch_size)
        y, _ = self._forward_flow(z)
        return torch.clamp(y, -1.0, 1.0)

    @staticmethod
    def discretize(y: torch.Tensor) -> torch.Tensor:
        """Discretise continuous outputs to binary {0, 1} by thresholding.

        Values at or above zero are mapped to 1; values below zero are
        mapped to 0.

        Parameters
        ----------
        y : torch.Tensor
            Continuous tensor, shape ``(..., num_sites)``.

        Returns
        -------
        torch.Tensor
            Binary tensor with values in ``{0, 1}``, same shape as *y*,
            dtype ``torch.float32``.
        """
        return (y >= 0.0).float()

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample discrete binary configurations.

        Draws continuous samples from the flow, discretises them, and
        returns both the full batch and the unique configurations.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.

        Returns
        -------
        configs : torch.Tensor
            All discrete configurations, shape ``(batch_size, num_sites)``.
        unique_configs : torch.Tensor
            Unique configurations, shape ``(n_unique, num_sites)`` where
            ``n_unique <= batch_size``.
        """
        y = self.sample_continuous(batch_size)
        configs = self.discretize(y)
        unique_configs = torch.unique(configs, dim=0)
        return configs, unique_configs

    def log_prob_continuous(self, y: torch.Tensor) -> torch.Tensor:
        """Compute log-probability in continuous space via change of variables.

        Uses the inverse flow to map data-space samples back to the
        prior, then applies the change-of-variables formula:
        ``log p(y) = log p_prior(z) + log |det J^{-1}|``.

        Parameters
        ----------
        y : torch.Tensor
            Continuous data-space samples, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities, shape ``(batch,)``.
        """
        z, log_det_inv = self._inverse_flow(y)
        self.prior.device = y.device
        log_pz = self.prior.log_prob(z)
        return log_pz + log_det_inv

    def log_prob_discrete(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate discrete log-probability via Monte Carlo integration.

        For each discrete configuration ``x`` in ``{0, 1}^n``, the
        probability is estimated by integrating the continuous density
        over the corresponding Voronoi cell (``[-1, 0)`` for 0 and
        ``[0, 1]`` for 1).  This is done by sampling uniform noise
        within each cell and averaging the continuous density.

        Parameters
        ----------
        x : torch.Tensor
            Discrete configurations with values in ``{0, 1}``, shape
            ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Estimated log-probabilities, shape ``(batch,)``.
        """
        batch = x.shape[0]
        device = x.device
        n_mc = self.n_mc_samples

        # Expand x for MC samples: (batch, n_mc, num_sites)
        x_expanded = x.unsqueeze(1).expand(batch, n_mc, self.num_sites)

        # Uniform noise within the Voronoi cell of each discrete value
        # site == 0 -> sample from [-1, 0), site == 1 -> sample from [0, 1]
        noise = torch.rand(batch, n_mc, self.num_sites, device=device)
        y_mc = torch.where(
            x_expanded == 1.0,
            noise,                  # [0, 1] for site == 1
            noise - 1.0,            # [-1, 0) for site == 0
        )

        # Flatten for log_prob_continuous: (batch * n_mc, num_sites)
        y_flat = y_mc.reshape(batch * n_mc, self.num_sites)
        log_probs = self.log_prob_continuous(y_flat)
        log_probs = log_probs.reshape(batch, n_mc)

        # Monte Carlo estimate: log E[p(y)] = logsumexp(log p(y)) - log(n_mc)
        log_prob_est = torch.logsumexp(log_probs, dim=1) - math.log(n_mc)
        return log_prob_est

    def forward(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: sample and compute log-probabilities.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.

        Returns
        -------
        configs : torch.Tensor
            Discrete configurations, shape ``(batch_size, num_sites)``.
        unique_configs : torch.Tensor
            Unique configurations, shape ``(n_unique, num_sites)``.
        log_probs : torch.Tensor
            Continuous log-probabilities at the pre-discretisation points,
            shape ``(batch_size,)``.
        """
        device = next(self.parameters()).device
        self.prior.device = device
        z = self.prior.sample(batch_size)
        y, log_det = self._forward_flow(z)
        y_clamped = torch.clamp(y, -1.0, 1.0)

        configs = self.discretize(y_clamped)
        unique_configs = torch.unique(configs, dim=0)

        log_pz = self.prior.log_prob(z)
        log_probs = log_pz + log_det

        return configs, unique_configs, log_probs
