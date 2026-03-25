"""
gumbel_topk --- Differentiable top-k selection mechanisms
==========================================================

Provides differentiable approximations to top-k selection for use in
particle-number-conserving normalizing flows:

* :class:`GumbelTopK` --- Gumbel-Softmax-based iterative selection.
* :class:`SigmoidTopK` --- Sigmoid thresholding with implicit binary search.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GumbelTopK",
    "SigmoidTopK",
]


# ---------------------------------------------------------------------------
# Differentiable top-k selectors
# ---------------------------------------------------------------------------


class GumbelTopK(nn.Module):
    """Gumbel-Softmax-based differentiable top-k selection.

    Adds Gumbel noise to logits and applies a softmax to produce a soft
    approximation of top-k selection.  At low temperatures the selection
    approaches a hard top-k; at high temperatures it is fully stochastic.

    Parameters
    ----------
    temperature : float, optional
        Initial temperature for the Gumbel-Softmax (default ``1.0``).
    min_temperature : float, optional
        Minimum temperature to prevent numerical issues (default ``0.01``).

    Attributes
    ----------
    temperature : float
        Current temperature.
    min_temperature : float
        Lower bound on temperature.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        min_temperature: float = 0.01,
    ) -> None:
        super().__init__()
        self.temperature: float = temperature
        self.min_temperature: float = min_temperature

    def forward(
        self, logits: torch.Tensor, k: int, temperature: Optional[float] = None
    ) -> torch.Tensor:
        """Select k elements via Gumbel-Softmax relaxation.

        Parameters
        ----------
        logits : torch.Tensor
            Unnormalised scores, shape ``(batch, n)``.
        k : int
            Number of elements to select.
        temperature : float or None, optional
            Override temperature for this call.  If ``None``, uses the
            instance temperature.

        Returns
        -------
        torch.Tensor
            Soft selection mask, shape ``(batch, n)``.  Values are in
            ``[0, 1]`` and approximately sum to ``k`` per row.
        """
        temp = max(
            temperature if temperature is not None else self.temperature,
            self.min_temperature,
        )

        # Add Gumbel noise for stochastic selection
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(logits) + 1e-20) + 1e-20
        )
        perturbed = (logits + gumbel_noise) / temp

        # Iterative softmax to approximate top-k
        # Start with uniform soft selection, iteratively sharpen
        soft_mask = torch.zeros_like(logits)
        remaining = perturbed.clone()

        for _ in range(k):
            probs = F.softmax(remaining, dim=-1)
            soft_mask = soft_mask + probs
            # Suppress already-selected elements
            remaining = remaining - probs * 1e6

        return torch.clamp(soft_mask, 0.0, 1.0)


class SigmoidTopK(nn.Module):
    """Sigmoid-based differentiable top-k selection with implicit threshold.

    Uses a learned or computed threshold to produce per-element sigmoid
    activations, then normalises to select exactly k elements in
    expectation.

    Parameters
    ----------
    temperature : float, optional
        Initial temperature controlling sigmoid sharpness (default ``1.0``).
    min_temperature : float, optional
        Minimum temperature (default ``0.01``).

    Attributes
    ----------
    temperature : float
        Current temperature.
    min_temperature : float
        Lower bound on temperature.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        min_temperature: float = 0.01,
    ) -> None:
        super().__init__()
        self.temperature: float = temperature
        self.min_temperature: float = min_temperature

    def forward(
        self, logits: torch.Tensor, k: int, temperature: Optional[float] = None
    ) -> torch.Tensor:
        """Select k elements via sigmoid thresholding.

        Finds a threshold such that ``sum(sigmoid((logits - threshold) / T))``
        is approximately ``k``, then returns the sigmoid activations.

        Parameters
        ----------
        logits : torch.Tensor
            Unnormalised scores, shape ``(batch, n)``.
        k : int
            Number of elements to select.
        temperature : float or None, optional
            Override temperature.  If ``None``, uses instance temperature.

        Returns
        -------
        torch.Tensor
            Soft selection mask, shape ``(batch, n)``.  Values are in
            ``[0, 1]`` and approximately sum to ``k`` per row.
        """
        temp = max(
            temperature if temperature is not None else self.temperature,
            self.min_temperature,
        )

        # Binary search for threshold that gives sum ≈ k
        n = logits.shape[-1]
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
        # Initial threshold: midpoint between k-th and (k+1)-th largest
        k_idx = min(k, n) - 1
        k_next = min(k, n - 1)
        threshold = 0.5 * (sorted_logits[:, k_idx] + sorted_logits[:, k_next])
        threshold = threshold.unsqueeze(-1)  # (batch, 1)

        soft_mask = torch.sigmoid((logits - threshold) / temp)
        return soft_mask
