"""
trotter_sampler --- Trotterized time-evolution configuration sampler
====================================================================

Implements :class:`TrotterSampler`, which performs classical Trotter
simulation of real-time evolution under a Hamiltonian and samples
computational-basis configurations from the evolved state.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.sparse.linalg import expm_multiply

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.samplers.sampler import Sampler, SamplerResult

__all__ = [
    "TrotterSampler",
]

logger = logging.getLogger(__name__)


class TrotterSampler(Sampler):
    """Trotterized time-evolution sampler.

    Evolves an initial state under the Hamiltonian via
    :math:`|\\psi(t)\\rangle = e^{-iHt} |\\psi_0\\rangle` and samples
    computational-basis configurations from the Born-rule distribution
    :math:`p(x) = |\\langle x | \\psi(t) \\rangle|^2`.

    Uses :func:`scipy.sparse.linalg.expm_multiply` for the matrix
    exponential, which is efficient for sparse Hamiltonians.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The system Hamiltonian.
    n_trotter_steps : int, optional
        Number of Trotter steps in the time evolution (default ``10``).
    time_step : float, optional
        Time step per Trotter step (default ``0.1``).
    initial_state : np.ndarray or None, optional
        Initial state vector.  If ``None``, uses the first computational-
        basis state (index 0).

    Attributes
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian.
    n_trotter_steps : int
        Number of Trotter steps.
    time_step : float
        Time step per step.
    total_time : float
        Total evolution time ``n_trotter_steps * time_step``.

    Examples
    --------
    >>> sampler = TrotterSampler(hamiltonian, n_trotter_steps=5)
    >>> result = sampler.sample(1000)
    >>> result.configs.shape[0]
    1000
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        n_trotter_steps: int = 10,
        time_step: float = 0.1,
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        if n_trotter_steps < 1:
            raise ValueError(
                f"n_trotter_steps must be >= 1, got {n_trotter_steps}"
            )
        if time_step <= 0.0:
            raise ValueError(f"time_step must be > 0, got {time_step}")

        self.hamiltonian: Hamiltonian = hamiltonian
        self.n_trotter_steps: int = n_trotter_steps
        self.time_step: float = time_step
        self.total_time: float = n_trotter_steps * time_step

        # Build dense Hamiltonian for expm_multiply
        self._h_dense = hamiltonian.to_dense().detach().cpu().numpy().astype(np.float64)
        self._dim = hamiltonian.hilbert_dim

        if initial_state is not None:
            if initial_state.shape[0] != self._dim:
                raise ValueError(
                    f"initial_state dimension ({initial_state.shape[0]}) does "
                    f"not match Hilbert-space dimension ({self._dim})"
                )
            self._initial_state = initial_state.astype(np.complex128)
        else:
            self._initial_state = self._default_initial_state()

    def _default_initial_state(self) -> np.ndarray:
        """Construct the default initial state (first basis vector).

        Returns
        -------
        np.ndarray
            State vector of shape ``(hilbert_dim,)`` with ``state[0] = 1``.
        """
        state = np.zeros(self._dim, dtype=np.complex128)
        state[0] = 1.0
        return state

    def _evolve(self) -> np.ndarray:
        """Evolve the initial state to the total evolution time.

        Returns
        -------
        np.ndarray
            Evolved state vector, shape ``(hilbert_dim,)``.
        """
        state = expm_multiply(
            -1j * self._h_dense,
            self._initial_state,
            start=0.0,
            stop=self.total_time,
            num=2,
            endpoint=True,
        )
        return state[-1]

    def sample(self, n_samples: int) -> SamplerResult:
        """Sample configurations from the evolved state.

        Parameters
        ----------
        n_samples : int
            Number of configuration samples to draw.

        Returns
        -------
        SamplerResult
            Sampled configurations with bitstring counts and metadata.

        Raises
        ------
        ValueError
            If ``n_samples < 1``.
        RuntimeError
            If the evolved state has near-zero norm.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        t_start = time.perf_counter()

        # Evolve
        evolved_state = self._evolve()

        # Compute Born-rule probabilities
        probabilities = np.abs(evolved_state) ** 2
        prob_sum = probabilities.sum()
        if prob_sum < 1e-15:
            raise RuntimeError(
                "Evolved state has near-zero norm; cannot sample."
            )
        probabilities = probabilities / prob_sum

        # Sample indices
        rng = np.random.default_rng()
        indices = rng.choice(self._dim, size=n_samples, p=probabilities)

        # Convert indices to binary configurations
        configs_list = []
        for idx in indices:
            config = self.hamiltonian._index_to_config(int(idx))
            configs_list.append(config)
        configs = torch.stack(configs_list)

        # Build counts
        bitstrings = [
            "".join(str(int(b)) for b in row) for row in configs.int()
        ]
        counts = dict(Counter(bitstrings))

        unique_indices = np.unique(indices)
        sample_time = time.perf_counter() - t_start

        metadata: Dict[str, Any] = {
            "n_unique": len(unique_indices),
            "unique_ratio": len(unique_indices) / max(n_samples, 1),
            "sample_time": sample_time,
            "total_evolution_time": self.total_time,
            "n_trotter_steps": self.n_trotter_steps,
            "time_step": self.time_step,
        }

        logger.info(
            "TrotterSampler: drew %d samples (%d unique) in %.3fs, "
            "total_time=%.2f",
            n_samples,
            len(unique_indices),
            sample_time,
            self.total_time,
        )

        return SamplerResult(
            configs=configs,
            counts=counts,
            metadata=metadata,
        )
