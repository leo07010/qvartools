"""
nf_sampler --- Normalizing-flow configuration sampler
=====================================================

Implements :class:`NFSampler`, which wraps a trained normalizing flow
(and optionally an NQS) to generate computational-basis configurations
for sample-based quantum diagonalization.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.samplers.sampler import Sampler, SamplerResult

__all__ = [
    "NFSampler",
]

logger = logging.getLogger(__name__)


class NFSampler(Sampler):
    """Normalizing-flow-based configuration sampler.

    Wraps a trained normalizing flow to generate discrete binary
    configurations.  Optionally uses an NQS for importance weighting
    or a Hamiltonian for local-energy estimation.

    Parameters
    ----------
    flow : nn.Module
        A trained normalizing-flow model.  Must implement
        ``sample(batch_size)`` returning ``(all_configs, unique_configs)``.
    nqs : nn.Module or None, optional
        A neural quantum state for importance weighting.  If provided,
        configurations are weighted by ``|psi(x)|^2``.
    hamiltonian : Hamiltonian or None, optional
        The Hamiltonian for energy-based metadata.  If provided, the
        sampler computes local energy statistics.
    device : str, optional
        Torch device for sampling (default ``"cpu"``).

    Attributes
    ----------
    flow : nn.Module
        The flow model.
    nqs : nn.Module or None
        The NQS model (optional).
    hamiltonian : Hamiltonian or None
        The Hamiltonian (optional).
    device : torch.device
        Active device.

    Examples
    --------
    >>> sampler = NFSampler(flow=trained_flow)
    >>> result = sampler.sample(1000)
    >>> result.configs.shape
    torch.Size([1000, 10])
    """

    def __init__(
        self,
        flow: nn.Module,
        nqs: Optional[nn.Module] = None,
        hamiltonian: Optional[Hamiltonian] = None,
        device: str = "cpu",
    ) -> None:
        self.flow: nn.Module = flow
        self.nqs: Optional[nn.Module] = nqs
        self.hamiltonian: Optional[Hamiltonian] = hamiltonian
        self.device: torch.device = torch.device(device)

        self.flow = self.flow.to(self.device)
        if self.nqs is not None:
            self.nqs = self.nqs.to(self.device)

    def sample(self, n_samples: int) -> SamplerResult:
        """Draw configuration samples from the trained flow.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        SamplerResult
            Sampled configurations with bitstring counts and metadata.

        Raises
        ------
        ValueError
            If ``n_samples < 1``.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        t_start = time.perf_counter()

        self.flow.eval()
        with torch.no_grad():
            all_configs, unique_configs = self.flow.sample(n_samples)

        all_configs = all_configs.to(self.device)

        # Build bitstring counts
        counts = self._build_counts(all_configs)

        # Compute metadata
        n_unique = unique_configs.shape[0]
        sample_time = time.perf_counter() - t_start

        metadata: Dict[str, Any] = {
            "n_unique": n_unique,
            "unique_ratio": n_unique / max(n_samples, 1),
            "sample_time": sample_time,
        }

        # Optional NQS weighting
        if self.nqs is not None:
            with torch.no_grad():
                log_amp = self.nqs.log_amplitude(all_configs.float())
                log_prob = 2.0 * log_amp
                log_z = torch.logsumexp(log_prob, dim=0)
                weights = torch.exp(log_prob - log_z)
            metadata["nqs_weights_mean"] = float(weights.mean())
            metadata["nqs_weights_std"] = float(weights.std())

        logger.info(
            "NFSampler: drew %d samples (%d unique) in %.3fs",
            n_samples,
            n_unique,
            sample_time,
        )

        return SamplerResult(
            configs=all_configs,
            counts=counts,
            metadata=metadata,
        )

    @staticmethod
    def _build_counts(configs: torch.Tensor) -> Dict[str, int]:
        """Build bitstring occurrence counts from configuration tensor.

        Parameters
        ----------
        configs : torch.Tensor
            Configurations, shape ``(n, n_sites)``.

        Returns
        -------
        dict
            Mapping from bitstring to count.
        """
        bitstrings = [
            "".join(str(int(b)) for b in row) for row in configs.cpu().int()
        ]
        return dict(Counter(bitstrings))
