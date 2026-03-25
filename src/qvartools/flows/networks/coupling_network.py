"""
coupling_network --- Multi-modal prior and RealNVP coupling layers
==================================================================

Provides the building blocks for the discrete normalizing flow:

* :class:`MultiModalPrior` --- mixture-of-Gaussians prior centred at
  +/- 1, ensuring uniform coverage of both ``{0, 1}`` values after
  discretisation.
* :class:`CouplingNetwork` --- MLP that produces scale and shift
  parameters for RealNVP affine coupling layers.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

__all__ = [
    "MultiModalPrior",
    "CouplingNetwork",
]


# ---------------------------------------------------------------------------
# Multi-modal prior
# ---------------------------------------------------------------------------


class MultiModalPrior:
    """Mixture-of-Gaussians prior centred at +/-1 for each site.

    Each site has an equal-weight mixture of two Gaussians,
    ``N(-1, std^2)`` and ``N(+1, std^2)``, so that the prior naturally
    covers the region around both 0 and 1 after thresholding.

    Parameters
    ----------
    num_sites : int
        Dimensionality of the latent space.
    std : float, optional
        Standard deviation of each Gaussian component (default ``1.0``).
    device : str, optional
        Torch device (default ``"cpu"``).

    Attributes
    ----------
    num_sites : int
        Number of sites / latent dimensions.
    std : float
        Component standard deviation.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        num_sites: int,
        std: float = 1.0,
        device: str = "cpu",
    ) -> None:
        if num_sites < 1:
            raise ValueError(f"num_sites must be >= 1, got {num_sites}")
        if std <= 0.0:
            raise ValueError(f"std must be > 0, got {std}")
        self.num_sites: int = num_sites
        self.std: float = std
        self.device: str = device

    def sample(self, batch_size: int) -> torch.Tensor:
        """Draw samples from the mixture-of-Gaussians prior.

        For each sample and each site, a component is selected uniformly
        at random (centred at -1 or +1), then a Gaussian sample is drawn
        from that component.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.

        Returns
        -------
        torch.Tensor
            Samples of shape ``(batch_size, num_sites)``.
        """
        # Choose component: 0 -> centre at -1, 1 -> centre at +1
        components = torch.randint(
            0, 2, (batch_size, self.num_sites), device=self.device
        ).float()
        centres = 2.0 * components - 1.0  # maps {0,1} -> {-1,+1}
        noise = torch.randn(
            batch_size, self.num_sites, device=self.device
        ) * self.std
        return centres + noise

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the log-probability under the mixture prior.

        The per-site log-probability is
        ``log(0.5 * N(z; -1, std^2) + 0.5 * N(z; +1, std^2))``,
        and the full log-probability is the sum over sites.

        Parameters
        ----------
        z : torch.Tensor
            Points at which to evaluate, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities, shape ``(batch,)``.
        """
        log_norm = -0.5 * math.log(2.0 * math.pi) - math.log(self.std)
        var = self.std ** 2

        # Log-prob for each component: shape (batch, num_sites)
        log_p_neg = log_norm - 0.5 * (z + 1.0) ** 2 / var
        log_p_pos = log_norm - 0.5 * (z - 1.0) ** 2 / var

        # log(0.5 * exp(a) + 0.5 * exp(b)) = log(0.5) + logsumexp(a, b)
        log_mix = math.log(0.5) + torch.logaddexp(log_p_neg, log_p_pos)

        return log_mix.sum(dim=-1)


# ---------------------------------------------------------------------------
# Coupling network for RealNVP
# ---------------------------------------------------------------------------


class CouplingNetwork(nn.Module):
    """MLP for RealNVP affine coupling layers.

    Maps the masked portion of the input to scale and shift parameters
    for the unmasked portion.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the masked input (number of masked sites).
    hidden_dims : list of int
        Hidden-layer sizes.
    output_dim : int
        Dimensionality of the output (number of unmasked sites).
        The network produces ``2 * output_dim`` values: the first half
        is the scale (passed through tanh), the second half is the shift.

    Attributes
    ----------
    net : nn.Sequential
        The underlying MLP.
    output_dim : int
        Number of unmasked sites.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.output_dim: int = output_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LeakyReLU(0.01))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2 * output_dim))

        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(
        self, masked_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and shift from the masked input.

        Parameters
        ----------
        masked_x : torch.Tensor
            The masked portion of the input, shape ``(batch, input_dim)``.

        Returns
        -------
        scale : torch.Tensor
            Scale parameters (tanh-bounded), shape ``(batch, output_dim)``.
        shift : torch.Tensor
            Shift parameters, shape ``(batch, output_dim)``.
        """
        out = self.net(masked_x)
        raw_scale = out[:, : self.output_dim]
        shift = out[:, self.output_dim :]
        scale = torch.tanh(raw_scale)
        return scale, shift
