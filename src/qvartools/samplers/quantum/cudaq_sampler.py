"""
cudaq_sampler --- CUDA-Q UCCSD circuit sampler
==============================================

Implements :class:`CUDAQCircuitSampler`, which uses a particle-number-
conserving UCCSD ansatz built from Givens rotations on the CUDA-Q
platform.  All samples satisfy the correct electron count by
construction -- no post-selection is needed.

Requires the ``cudaq`` package (optional dependency).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from qvartools.samplers.sampler import Sampler, SamplerResult

__all__ = [
    "CUDAQSamplerConfig",
    "CUDAQCircuitSampler",
]

logger = logging.getLogger(__name__)

try:
    import cudaq

    _HAS_CUDAQ = True
except ImportError:
    _HAS_CUDAQ = False


@dataclass(frozen=True)
class CUDAQSamplerConfig:
    """Configuration for :class:`CUDAQCircuitSampler`.

    Parameters
    ----------
    n_layers : int
        Number of brick-wall UCCSD layers (default ``2``).
    shots : int
        Number of circuit shots per sampling call (default ``10000``).
    target : str
        CUDA-Q execution target (default ``"qpp-cpu"``).
    """

    n_layers: int = 2
    shots: int = 10000
    target: str = "qpp-cpu"


class CUDAQCircuitSampler(Sampler):
    """CUDA-Q UCCSD circuit sampler.

    Particle-number-conserving: all samples have exactly
    ``n_alpha + n_beta`` electrons.  No post-selection is needed.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian providing ``n_orbitals``, ``n_alpha``,
        and ``n_beta``.
    config : CUDAQSamplerConfig or None, optional
        Sampler configuration.  Uses defaults when ``None``.

    Raises
    ------
    ImportError
        If the ``cudaq`` package is not installed.

    Examples
    --------
    >>> sampler = CUDAQCircuitSampler(hamiltonian)
    >>> result = sampler.sample(5000)
    >>> result.configs.shape[1]  # == 2 * n_orbitals
    10
    """

    def __init__(
        self,
        hamiltonian,
        config: Optional[CUDAQSamplerConfig] = None,
    ) -> None:
        if not _HAS_CUDAQ:
            raise ImportError(
                "cudaq is required for CUDAQCircuitSampler. "
                "Install with: pip install cuda-quantum"
            )

        from qvartools.samplers.quantum.cudaq_circuits import count_uccsd_params

        self.hamiltonian = hamiltonian
        self.config: CUDAQSamplerConfig = config or CUDAQSamplerConfig()

        self.n_orbitals: int = hamiltonian.n_orbitals
        self.n_alpha: int = hamiltonian.n_alpha
        self.n_beta: int = hamiltonian.n_beta
        self.n_qubits: int = 2 * self.n_orbitals

        cudaq.set_target(self.config.target)

        self.n_params: int = count_uccsd_params(
            self.n_orbitals, self.config.n_layers
        )
        self._params: np.ndarray = np.random.randn(self.n_params) * 0.01

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def set_params(self, params) -> None:
        """Set the variational parameters.

        Parameters
        ----------
        params : array-like
            New parameter vector of length :attr:`n_params`.
        """
        self._params = np.array(params, dtype=np.float64)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples: int) -> SamplerResult:
        """Draw configuration samples from the UCCSD circuit.

        Parameters
        ----------
        n_samples : int
            Number of shots to execute.

        Returns
        -------
        SamplerResult
            Unique sampled configurations with metadata.

        Raises
        ------
        ValueError
            If ``n_samples < 1``.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        from qvartools.samplers.quantum.cudaq_circuits import uccsd_ansatz

        t_start = time.perf_counter()

        thetas = self._params.tolist()

        result = cudaq.sample(
            uccsd_ansatz,
            self.n_qubits,
            self.n_alpha,
            self.n_beta,
            self.config.n_layers,
            thetas,
            shots_count=n_samples,
        )

        # Convert bitstrings to configuration tensors
        configs_list: list[torch.Tensor] = []
        for bitstring in result:
            count = result.count(bitstring)
            config = torch.tensor(
                [int(b) for b in bitstring], dtype=torch.long
            )
            configs_list.extend([config] * count)

        wall_time = time.perf_counter() - t_start

        if not configs_list:
            logger.warning("CUDAQCircuitSampler: no samples returned.")
            return SamplerResult(
                configs=torch.empty(0, self.n_qubits, dtype=torch.long),
                log_probs=None,
                wall_time=wall_time,
                metadata={"error": "No samples"},
            )

        configs = torch.stack(configs_list)
        unique_configs = torch.unique(configs, dim=0)

        # Build bitstring counts
        bitstrings = [
            "".join(str(int(b)) for b in row) for row in configs.int()
        ]
        counts: Dict[str, int] = dict(Counter(bitstrings))

        metadata: Dict[str, Any] = {
            "n_raw_samples": n_samples,
            "n_unique": len(unique_configs),
            "unique_ratio": len(unique_configs) / max(n_samples, 1),
            "n_params": self.n_params,
            "sampler_type": "CUDAQ-UCCSD",
        }

        logger.info(
            "CUDAQCircuitSampler: drew %d shots (%d unique) in %.3fs",
            n_samples,
            len(unique_configs),
            wall_time,
        )

        return SamplerResult(
            configs=unique_configs,
            counts=counts,
            log_probs=None,
            wall_time=wall_time,
            metadata=metadata,
        )
