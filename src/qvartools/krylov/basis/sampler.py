"""
sampler --- Krylov basis sampler
================================

Provides a unified interface for sampling computational-basis configurations
from Krylov states, supporting both classical simulation and (future)
quantum-hardware backends.

The sampler computes :math:`e^{-iH\\Delta t \\cdot k}|\\psi_0\\rangle` and
samples bitstring configurations from the resulting probability distribution.

Classes
-------
KrylovBasisSampler
    Interface for sampling configurations from Krylov-evolved states.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from scipy.sparse.linalg import expm_multiply

from qvartools.hamiltonians.hamiltonian import Hamiltonian

__all__ = [
    "KrylovBasisSampler",
]

logger = logging.getLogger(__name__)


class KrylovBasisSampler:
    r"""Interface for sampling configurations from Krylov-evolved states.

    Computes time-evolved states
    :math:`|\psi_k\rangle = e^{-iH \Delta t \cdot k} |\psi_0\rangle`
    and samples computational-basis configurations from the resulting
    probability distribution :math:`p(x) = |\langle x | \psi_k \rangle|^2`.

    Currently provides a classical simulation backend.  The interface is
    designed to accommodate future quantum-hardware backends (e.g., via
    Qiskit or CUDA-Q) without changing the calling code.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The system Hamiltonian.
    num_qubits : int
        Number of qubits (must match ``hamiltonian.num_sites``).
    shots : int, optional
        Default number of measurement shots per sample call.
    time_step : float, optional
        Time step :math:`\Delta t` for the Krylov evolution.

    Raises
    ------
    ValueError
        If ``num_qubits`` does not match the Hamiltonian's site count,
        or if ``shots`` or ``time_step`` are invalid.

    Examples
    --------
    >>> from qvartools.hamiltonians import TransverseFieldIsing
    >>> ham = TransverseFieldIsing(num_sites=4, J=1.0, h=0.5)
    >>> sampler = KrylovBasisSampler(ham, num_qubits=4, shots=500)
    >>> counts = sampler.sample_krylov_state(krylov_power=3)
    >>> type(counts)
    <class 'dict'>
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        num_qubits: int,
        shots: int = 1000,
        time_step: float = 0.1,
    ) -> None:
        if num_qubits != hamiltonian.num_sites:
            raise ValueError(
                f"num_qubits ({num_qubits}) does not match "
                f"hamiltonian.num_sites ({hamiltonian.num_sites})"
            )
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")
        if time_step <= 0.0:
            raise ValueError(f"time_step must be > 0, got {time_step}")

        self.hamiltonian = hamiltonian
        self.num_qubits = num_qubits
        self.shots = shots
        self.time_step = time_step

        # Pre-build dense Hamiltonian for classical simulation
        self._h_dense: Optional[np.ndarray] = None

    @property
    def h_dense(self) -> np.ndarray:
        """Lazily-constructed dense Hamiltonian matrix.

        Returns
        -------
        np.ndarray
            Dense Hamiltonian of shape ``(hilbert_dim, hilbert_dim)``.
        """
        if self._h_dense is None:
            h_torch = self.hamiltonian.to_dense()
            self._h_dense = h_torch.detach().cpu().numpy().astype(np.float64)
        return self._h_dense

    def sample_krylov_state(
        self,
        krylov_power: int,
        initial_state: Optional[np.ndarray] = None,
    ) -> Dict[str, int]:
        r"""Sample configurations from a Krylov-evolved state.

        Computes :math:`|\psi_k\rangle = e^{-iH \Delta t \cdot k} |\psi_0\rangle`
        and samples ``shots`` bitstring configurations from the resulting
        probability distribution.

        Parameters
        ----------
        krylov_power : int
            Krylov power index :math:`k`.  ``k = 0`` samples from the
            initial state directly.
        initial_state : np.ndarray or None, optional
            Initial state vector of shape ``(hilbert_dim,)``.  If ``None``,
            the first computational-basis state ``|00...0\rangle`` is used.

        Returns
        -------
        dict of str to int
            Mapping from bitstring (e.g., ``"0110"``) to the number of
            times it was observed across ``shots`` samples.

        Raises
        ------
        ValueError
            If ``krylov_power`` is negative or ``initial_state`` has the
            wrong dimension.
        RuntimeError
            If the evolved state has near-zero norm (e.g., due to
            numerical issues).
        """
        if krylov_power < 0:
            raise ValueError(
                f"krylov_power must be >= 0, got {krylov_power}"
            )

        hilbert_dim = self.hamiltonian.hilbert_dim

        if initial_state is not None:
            if initial_state.shape[0] != hilbert_dim:
                raise ValueError(
                    f"initial_state dimension ({initial_state.shape[0]}) does "
                    f"not match Hilbert-space dimension ({hilbert_dim})"
                )
            psi0 = initial_state.astype(np.complex128)
        else:
            psi0 = np.zeros(hilbert_dim, dtype=np.complex128)
            psi0[0] = 1.0

        return self._sample_classical(krylov_power, psi0)

    def _sample_classical(
        self,
        krylov_power: int,
        initial_state: np.ndarray,
    ) -> Dict[str, int]:
        r"""Classical simulation backend for Krylov sampling.

        Parameters
        ----------
        krylov_power : int
            Krylov power index :math:`k`.
        initial_state : np.ndarray
            Initial state vector, shape ``(hilbert_dim,)``.

        Returns
        -------
        dict of str to int
            Bitstring counts dictionary.

        Raises
        ------
        RuntimeError
            If the evolved state has near-zero norm.
        """
        if krylov_power == 0:
            state = initial_state.copy()
        else:
            total_time = krylov_power * self.time_step
            evolved = expm_multiply(
                -1j * self.h_dense,
                initial_state,
                start=0.0,
                stop=total_time,
                num=2,
                endpoint=True,
            )
            state = evolved[-1]

        # Compute probabilities
        probabilities = np.abs(state) ** 2
        prob_sum = probabilities.sum()
        if prob_sum < 1e-15:
            raise RuntimeError(
                "Evolved state has near-zero norm; cannot sample."
            )
        probabilities = probabilities / prob_sum

        # Sample indices
        rng = np.random.default_rng()
        sampled_indices = rng.choice(
            len(probabilities), size=self.shots, p=probabilities
        )

        # Convert to bitstring counts
        counts: Dict[str, int] = {}
        for idx in sampled_indices:
            bitstring = format(int(idx), f"0{self.num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts
