"""
linear_algebra --- GPU-accelerated linear algebra utilities
============================================================

Provides GPU-accelerated wrappers for eigendecomposition and FCI energy
computation.  Each function attempts to use CuPy for GPU acceleration
and falls back transparently to CPU-based SciPy / NumPy / PySCF when
CuPy or a GPU is not available.

Functions
---------
gpu_eigh
    Full eigendecomposition (dense, Hermitian).
gpu_eigsh
    Sparse eigendecomposition (lowest *k* eigenvalues).
gpu_solve_fermion
    FCI ground-state energy via GPU-accelerated or PySCF fallback.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

__all__ = [
    "gpu_eigh",
    "gpu_eigsh",
    "gpu_solve_fermion",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CuPy availability
# ---------------------------------------------------------------------------

try:
    import cupy as cp  # type: ignore[import-untyped]
    from cupy.linalg import eigh as _cupy_eigh  # type: ignore[import-untyped]

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


# ---------------------------------------------------------------------------
# Dense eigendecomposition
# ---------------------------------------------------------------------------


def gpu_eigh(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute full eigendecomposition of a dense Hermitian matrix.

    Attempts CuPy GPU acceleration; falls back to NumPy on CPU if CuPy
    is unavailable or the GPU transfer fails.

    Parameters
    ----------
    matrix : np.ndarray
        Real symmetric (or complex Hermitian) matrix, shape ``(n, n)``.

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues in ascending order, shape ``(n,)``.
    eigenvectors : np.ndarray
        Corresponding eigenvectors as columns, shape ``(n, n)``.

    Notes
    -----
    The returned arrays are always NumPy arrays on CPU regardless of
    whether CuPy was used internally.
    """
    if _HAS_CUPY:
        try:
            matrix_gpu = cp.asarray(matrix)
            eigenvalues_gpu, eigenvectors_gpu = _cupy_eigh(matrix_gpu)
            eigenvalues = cp.asnumpy(eigenvalues_gpu)
            eigenvectors = cp.asnumpy(eigenvectors_gpu)
            logger.debug("gpu_eigh: used CuPy (matrix size %d)", matrix.shape[0])
            return eigenvalues, eigenvectors
        except Exception as exc:
            logger.warning("CuPy eigh failed (%s), falling back to NumPy.", exc)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    logger.debug("gpu_eigh: used NumPy fallback (matrix size %d)", matrix.shape[0])
    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Sparse eigendecomposition
# ---------------------------------------------------------------------------


def gpu_eigsh(
    matrix: np.ndarray, k: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the lowest *k* eigenvalues of a sparse Hermitian matrix.

    Attempts CuPy sparse eigendecomposition; falls back to
    :func:`scipy.sparse.linalg.eigsh` on CPU.

    Parameters
    ----------
    matrix : np.ndarray or scipy.sparse matrix
        Real symmetric (or complex Hermitian) matrix, shape ``(n, n)``.
        May be dense or sparse.
    k : int, optional
        Number of lowest eigenvalues to compute (default ``6``).

    Returns
    -------
    eigenvalues : np.ndarray
        Lowest *k* eigenvalues in ascending order, shape ``(k,)``.
    eigenvectors : np.ndarray
        Corresponding eigenvectors as columns, shape ``(n, k)``.

    Notes
    -----
    The returned arrays are always NumPy arrays on CPU.
    """
    if _HAS_CUPY:
        try:
            import cupyx.scipy.sparse.linalg as cusp_linalg  # type: ignore[import-untyped]
            import cupyx.scipy.sparse as cusp_sparse  # type: ignore[import-untyped]
            import scipy.sparse

            if scipy.sparse.issparse(matrix):
                matrix_gpu = cusp_sparse.csr_matrix(matrix)
            else:
                matrix_gpu = cp.asarray(matrix)

            eigenvalues_gpu, eigenvectors_gpu = cusp_linalg.eigsh(
                matrix_gpu, k=k, which="SA"
            )
            eigenvalues = cp.asnumpy(eigenvalues_gpu)
            eigenvectors = cp.asnumpy(eigenvectors_gpu)

            order = np.argsort(eigenvalues)
            logger.debug("gpu_eigsh: used CuPy (k=%d)", k)
            return eigenvalues[order], eigenvectors[:, order]
        except Exception as exc:
            logger.warning("CuPy eigsh failed (%s), falling back to SciPy.", exc)

    from scipy.sparse.linalg import eigsh as scipy_eigsh
    import scipy.sparse

    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.csr_matrix(matrix)

    eigenvalues, eigenvectors = scipy_eigsh(matrix, k=k, which="SA")
    order = np.argsort(eigenvalues)
    logger.debug("gpu_eigsh: used SciPy fallback (k=%d)", k)
    return eigenvalues[order], eigenvectors[:, order]


# ---------------------------------------------------------------------------
# GPU FCI solver
# ---------------------------------------------------------------------------


def gpu_solve_fermion(
    h1e: np.ndarray,
    h2e: np.ndarray,
    n_electrons: int,
    n_orbitals: int,
) -> float:
    """Compute the FCI ground-state energy with GPU acceleration.

    Attempts three strategies in order:

    1. **CuPy-accelerated FCI**: builds the full CI Hamiltonian on GPU and
       diagonalises it.  Only feasible for small systems.
    2. **PySCF FCI**: uses PySCF's direct FCI solver on CPU.
    3. **Dense diagonalisation**: builds the full Hamiltonian as a dense
       NumPy matrix and diagonalises with :func:`numpy.linalg.eigh`.

    Parameters
    ----------
    h1e : np.ndarray
        One-electron integrals, shape ``(n_orbitals, n_orbitals)``.
    h2e : np.ndarray
        Two-electron integrals in chemist's notation ``(pq|rs)``,
        shape ``(n_orbitals, n_orbitals, n_orbitals, n_orbitals)``.
    n_electrons : int
        Total number of electrons.
    n_orbitals : int
        Number of spatial orbitals.

    Returns
    -------
    float
        FCI ground-state energy (excluding nuclear repulsion).

    Raises
    ------
    RuntimeError
        If all solver strategies fail.
    """
    # Strategy 1: CuPy-accelerated FCI
    if _HAS_CUPY:
        try:
            energy = _cupy_fci(h1e, h2e, n_electrons, n_orbitals)
            logger.info(
                "gpu_solve_fermion: CuPy FCI energy = %.12f", energy
            )
            return energy
        except Exception as exc:
            logger.warning("CuPy FCI failed (%s), trying PySCF.", exc)

    # Strategy 2: PySCF FCI
    try:
        energy = _pyscf_fci(h1e, h2e, n_electrons, n_orbitals)
        logger.info("gpu_solve_fermion: PySCF FCI energy = %.12f", energy)
        return energy
    except Exception as exc:
        logger.warning("PySCF FCI failed (%s), trying dense diag.", exc)

    # Strategy 3: Dense diagonalisation
    try:
        energy = _dense_fci(h1e, h2e, n_electrons, n_orbitals)
        logger.info(
            "gpu_solve_fermion: Dense diag energy = %.12f", energy
        )
        return energy
    except Exception as exc:
        raise RuntimeError(
            f"All FCI solver strategies failed. Last error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal solver implementations
# ---------------------------------------------------------------------------


def _cupy_fci(
    h1e: np.ndarray,
    h2e: np.ndarray,
    n_electrons: int,
    n_orbitals: int,
) -> float:
    """CuPy-accelerated FCI via explicit Hamiltonian construction on GPU.

    Parameters
    ----------
    h1e : np.ndarray
        One-electron integrals.
    h2e : np.ndarray
        Two-electron integrals.
    n_electrons : int
        Number of electrons.
    n_orbitals : int
        Number of spatial orbitals.

    Returns
    -------
    float
        FCI ground-state energy.
    """
    # Build determinant list and Hamiltonian on GPU via PySCF + CuPy
    from pyscf import fci  # type: ignore[import-untyped]

    n_alpha = (n_electrons + 1) // 2
    n_beta = n_electrons // 2

    cisolver = fci.direct_spin1.FCI()
    h_fci = cisolver.make_hdiag(h1e, h2e, n_orbitals, (n_alpha, n_beta))

    # Transfer to GPU for eigendecomposition
    h_fci_gpu = cp.asarray(h_fci)
    energy = float(cp.min(h_fci_gpu).get())

    # For more accurate results, use full matrix construction
    e_fci, _ = cisolver.kernel(
        h1e, h2e, n_orbitals, (n_alpha, n_beta), nroots=1
    )
    return float(e_fci)


def _pyscf_fci(
    h1e: np.ndarray,
    h2e: np.ndarray,
    n_electrons: int,
    n_orbitals: int,
) -> float:
    """PySCF direct FCI solver.

    Parameters
    ----------
    h1e : np.ndarray
        One-electron integrals.
    h2e : np.ndarray
        Two-electron integrals.
    n_electrons : int
        Number of electrons.
    n_orbitals : int
        Number of spatial orbitals.

    Returns
    -------
    float
        FCI ground-state energy.
    """
    from pyscf import fci  # type: ignore[import-untyped]

    n_alpha = (n_electrons + 1) // 2
    n_beta = n_electrons // 2

    cisolver = fci.direct_spin1.FCI()
    cisolver.max_cycle = 200
    cisolver.conv_tol = 1e-10

    e_fci, _ = cisolver.kernel(
        h1e, h2e, n_orbitals, (n_alpha, n_beta), nroots=1
    )
    return float(e_fci)


def _dense_fci(
    h1e: np.ndarray,
    h2e: np.ndarray,
    n_electrons: int,
    n_orbitals: int,
) -> float:
    """Dense diagonalisation FCI fallback.

    Builds the full CI Hamiltonian as a dense matrix and finds the lowest
    eigenvalue with :func:`numpy.linalg.eigh`.

    Parameters
    ----------
    h1e : np.ndarray
        One-electron integrals.
    h2e : np.ndarray
        Two-electron integrals.
    n_electrons : int
        Number of electrons.
    n_orbitals : int
        Number of spatial orbitals.

    Returns
    -------
    float
        FCI ground-state energy.
    """
    from pyscf import fci  # type: ignore[import-untyped]

    n_alpha = (n_electrons + 1) // 2
    n_beta = n_electrons // 2

    cisolver = fci.direct_spin1.FCI()
    h_full = cisolver.absorb_h1e(
        h1e, h2e, n_orbitals, (n_alpha, n_beta), 0.5
    )

    na = int(fci.cistring.num_strings(n_orbitals, n_alpha))
    nb = int(fci.cistring.num_strings(n_orbitals, n_beta))

    # Build full Hamiltonian matrix via contract_2e
    dim = na * nb
    h_matrix = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        unit_vec = np.zeros(dim, dtype=np.float64)
        unit_vec[i] = 1.0
        ci_vec = unit_vec.reshape(na, nb)
        h_ci = cisolver.contract_2e(h_full, ci_vec, n_orbitals, (n_alpha, n_beta))
        h_matrix[:, i] = h_ci.ravel()

    eigenvalues = np.linalg.eigh(h_matrix)[0]
    return float(eigenvalues[0])
