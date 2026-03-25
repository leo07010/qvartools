"""
base --- Abstract sampler interface and result dataclass
========================================================

Defines the :class:`Sampler` ABC and the :class:`SamplerResult` immutable
dataclass that standardise sampler outputs across all sampling strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

__all__ = [
    "SamplerResult",
    "Sampler",
]


# ---------------------------------------------------------------------------
# SamplerResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerResult:
    """Immutable container for sampler output.

    Parameters
    ----------
    configs : torch.Tensor
        Sampled configurations, shape ``(n_samples, n_sites)``.  Each row
        is a computational-basis state with entries in ``{0, 1}``.
    counts : dict
        Mapping from bitstring representation to occurrence count.
        Keys are strings of ``'0'`` and ``'1'`` characters.
    metadata : dict
        Additional sampler-specific information (e.g. unique count,
        sampling time, flow parameters).
    log_probs : torch.Tensor or None, optional
        Log-probabilities of the sampled configurations, shape
        ``(n_samples,)``.  ``None`` when the sampler does not provide
        probabilities.
    wall_time : float, optional
        Wall-clock time in seconds spent generating the samples
        (default ``0.0``).

    Attributes
    ----------
    configs : torch.Tensor
    counts : dict
    metadata : dict
    log_probs : torch.Tensor or None
    wall_time : float

    Examples
    --------
    >>> result = SamplerResult(configs=torch.zeros(10, 4, dtype=torch.int64))
    >>> result.configs.shape
    torch.Size([10, 4])
    """

    configs: torch.Tensor
    counts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    log_probs: Optional[torch.Tensor] = None
    wall_time: float = 0.0


# ---------------------------------------------------------------------------
# Sampler ABC
# ---------------------------------------------------------------------------


class Sampler(ABC):
    """Abstract base class for all configuration samplers.

    Every subclass must implement :meth:`sample`, which draws a set of
    computational-basis configurations and returns a :class:`SamplerResult`.
    """

    @abstractmethod
    def sample(self, n_samples: int) -> SamplerResult:
        """Draw configuration samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        SamplerResult
            Sampled configurations, counts, and metadata.
        """
