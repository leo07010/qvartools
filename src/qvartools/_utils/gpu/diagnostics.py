"""
diagnostics --- GPU-accelerated diagonalization for projected Hamiltonians
==========================================================================

Drop-in replacement for ``qiskit_addon_sqd.fermion.solve_fermion``.
Uses ``torch.linalg.eigh`` for dense matrices, ``scipy.sparse.linalg.eigsh``
for larger ones, and CuPy sparse eigsh if available.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import torch
from scipy.sparse.linalg import eigsh as scipy_eigsh

__all__ = [
    "gpu_solve_fermion",
    "compute_occupancies",
]

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh

    try:
        cp.cuda.Device(0).compute_capability
        _CUPY_AVAILABLE = True
    except Exception:
        _CUPY_AVAILABLE = False
except ImportError:
    _CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

MAX_DENSE_CONFIGS = 10000
SPARSE_THRESHOLD = 3000
SPARSE_H_THRESHOLD = 8000


def compute_occupancies(
    configs: np.ndarray,
    v0: np.ndarray,
    n_orb: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute orbital occupancies from an eigenvector.

    Parameters
    ----------
    configs : np.ndarray
        ``(n_configs, 2*n_orb)`` occupation numbers.
    v0 : np.ndarray
        ``(n_configs,)`` eigenvector coefficients.
    n_orb : int or None
        Number of spatial orbitals. Inferred if ``None``.

    Returns
    -------
    tuple of np.ndarray
        ``(occ_alpha, occ_beta)`` each of shape ``(n_orb,)``.
    """
    configs_np = np.asarray(configs, dtype=np.float64)
    probs = np.abs(np.asarray(v0)) ** 2
    occ_flat = (probs[:, None] * configs_np).sum(axis=0)

    if n_orb is None:
        n_orb = configs_np.shape[1] // 2

    return (occ_flat[:n_orb], occ_flat[n_orb:])


def gpu_solve_fermion(
    configs: torch.Tensor | np.ndarray,
    hamiltonian: Any,
    max_dense: int = MAX_DENSE_CONFIGS,
) -> tuple[float, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """GPU-accelerated diagonalization in a configuration subspace.

    Parameters
    ----------
    configs : torch.Tensor or np.ndarray
        ``(n_configs, 2*n_orb)`` configurations.
    hamiltonian
        ``MolecularHamiltonian`` with ``matrix_elements_fast()`` and
        ``diagonal_element()`` methods.
    max_dense : int
        Maximum basis size for dense diagonalization.

    Returns
    -------
    energy : float
        Ground-state energy (total, including nuclear repulsion).
    eigenvector : np.ndarray
        Ground-state coefficients, shape ``(n_configs,)``.
    occupancies : tuple
        ``(occ_alpha, occ_beta)`` IBM-compatible occupancy format.
    """
    if isinstance(configs, np.ndarray):
        configs = torch.from_numpy(configs).long()
    else:
        configs = configs.detach().cpu().long()

    n = len(configs)
    n_orb = configs.shape[1] // 2

    if n == 0:
        raise ValueError("Empty basis — cannot diagonalize")

    if n == 1:
        e = float(hamiltonian.diagonal_element(configs[0]))
        v0 = np.array([1.0])
        occ = compute_occupancies(configs.numpy(), v0, n_orb)
        return e, v0, occ

    # Build projected Hamiltonian
    H_proj = hamiltonian.matrix_elements_fast(configs)
    H_np = H_proj.detach().cpu().numpy()
    if np.iscomplexobj(H_np):
        H_np = H_np.real
    H_np = H_np.astype(np.float64)
    H_np = 0.5 * (H_np + H_np.T)

    if n <= min(SPARSE_THRESHOLD, max_dense):
        E0, v0 = _dense_diag(H_np)
    else:
        E0, v0 = _iterative_diag(H_np)

    occ = compute_occupancies(configs.numpy(), v0, n_orb)
    return E0, v0, occ


def _dense_diag(H_np: np.ndarray) -> tuple[float, np.ndarray]:
    """Dense diagonalization via torch or numpy."""
    n = H_np.shape[0]

    if torch.cuda.is_available() and n <= 8000:
        try:
            H_gpu = torch.from_numpy(H_np).to("cuda")
            eigenvalues, eigenvectors = torch.linalg.eigh(H_gpu)
            return float(eigenvalues[0].cpu()), eigenvectors[:, 0].cpu().numpy()
        except Exception:
            pass

    eigenvalues, eigenvectors = np.linalg.eigh(H_np)
    return float(eigenvalues[0]), eigenvectors[:, 0]


def _iterative_diag(H_np: np.ndarray) -> tuple[float, np.ndarray]:
    """Iterative Lanczos diagonalization for large matrices."""
    n = H_np.shape[0]

    if _CUPY_AVAILABLE:
        try:
            H_gpu = cp.asarray(H_np)
            if n <= 8000:
                eigenvalues, eigenvectors = cp.linalg.eigh(H_gpu)
                E0 = float(cp.asnumpy(eigenvalues[0]))
                v0 = cp.asnumpy(eigenvectors[:, 0])
            else:
                H_sparse = cupy_csr(H_gpu)
                eigenvalues, eigenvectors = cupy_eigsh(H_sparse, k=1, which="SA")
                E0 = float(cp.asnumpy(eigenvalues[0]))
                v0 = cp.asnumpy(eigenvectors[:, 0])
            del H_gpu
            return E0, v0
        except Exception as e:
            warnings.warn(f"CuPy eigsh failed ({e}), falling back to SciPy")

    from scipy.sparse.linalg import LinearOperator

    matvec = lambda x: H_np @ x
    H_op = LinearOperator((n, n), matvec=matvec, dtype=H_np.dtype)
    eigenvalues, eigenvectors = scipy_eigsh(H_op, k=1, which="SA")
    return float(eigenvalues[0]), eigenvectors[:, 0]
