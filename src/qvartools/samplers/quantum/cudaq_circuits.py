"""
cudaq_circuits --- CUDA-Q quantum circuits for UCCSD-like ansatz
================================================================

Uses particle-number-conserving Givens rotations as building blocks.
Requires the ``cudaq`` package (optional dependency).
"""

from __future__ import annotations

__all__ = [
    "uccsd_ansatz",
    "count_uccsd_params",
]

try:
    import cudaq

    _HAS_CUDAQ = True
except ImportError:
    _HAS_CUDAQ = False


def count_uccsd_params(n_orbitals: int, n_layers: int) -> int:
    """Count parameters for the UCCSD ansatz.

    Parameters
    ----------
    n_orbitals : int
        Number of spatial orbitals.
    n_layers : int
        Number of brick-wall layers.

    Returns
    -------
    int
        Total number of variational parameters.
    """
    return n_layers * 2 * (n_orbitals - 1)


if _HAS_CUDAQ:

    @cudaq.kernel
    def uccsd_ansatz(
        n_qubits: int,
        n_alpha: int,
        n_beta: int,
        n_layers: int,
        thetas: list[float],
    ):
        """UCCSD-like ansatz using particle-number-conserving Givens rotations.

        Parameters
        ----------
        n_qubits : int
            Total number of qubits (``2 * n_orbitals``).
        n_alpha : int
            Number of alpha electrons.
        n_beta : int
            Number of beta electrons.
        n_layers : int
            Number of brick-wall layers.
        thetas : list[float]
            Variational rotation angles.
        """
        q = cudaq.qvector(n_qubits)
        n_orb = n_qubits // 2

        # Prepare Hartree-Fock reference
        for i in range(n_alpha):
            x(q[i])
        for i in range(n_beta):
            x(q[n_orb + i])

        param_idx = 0
        for layer in range(n_layers):
            # Alpha even pairs
            for i in range(0, n_orb - 1, 2):
                theta = thetas[param_idx]
                cx(q[i + 1], q[i])
                ry(theta, q[i + 1])
                cx(q[i], q[i + 1])
                ry(-theta, q[i + 1])
                cx(q[i], q[i + 1])
                cx(q[i + 1], q[i])
                param_idx += 1
            # Alpha odd pairs
            for i in range(1, n_orb - 1, 2):
                theta = thetas[param_idx]
                cx(q[i + 1], q[i])
                ry(theta, q[i + 1])
                cx(q[i], q[i + 1])
                ry(-theta, q[i + 1])
                cx(q[i], q[i + 1])
                cx(q[i + 1], q[i])
                param_idx += 1
            # Beta even pairs
            for i in range(0, n_orb - 1, 2):
                j = n_orb + i
                theta = thetas[param_idx]
                cx(q[j + 1], q[j])
                ry(theta, q[j + 1])
                cx(q[j], q[j + 1])
                ry(-theta, q[j + 1])
                cx(q[j], q[j + 1])
                cx(q[j + 1], q[j])
                param_idx += 1
            # Beta odd pairs
            for i in range(1, n_orb - 1, 2):
                j = n_orb + i
                theta = thetas[param_idx]
                cx(q[j + 1], q[j])
                ry(theta, q[j + 1])
                cx(q[j], q[j + 1])
                ry(-theta, q[j + 1])
                cx(q[j], q[j + 1])
                cx(q[j + 1], q[j])
                param_idx += 1

else:

    def uccsd_ansatz(*args, **kwargs):
        """Stub that raises when ``cudaq`` is not installed."""
        raise ImportError(
            "cudaq is required for uccsd_ansatz. "
            "Install with: pip install cuda-quantum"
        )
