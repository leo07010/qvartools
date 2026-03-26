"""
eigenvalue --- Eigenvalue problem solvers and overlap regularization
====================================================================

Provides functions for solving standard and generalized eigenvalue problems
arising from projected Hamiltonian construction.  Supports both CPU (SciPy)
and GPU (CuPy) backends.

Functions
---------
solve_generalized_eigenvalue
    Solve the generalized eigenvalue problem Hv = ESv.
compute_ground_state_energy
    Extract the ground-state energy from a Hamiltonian matrix.
analyze_spectrum
    Compute eigenvalues, gaps, and spectral statistics.
regularize_overlap_matrix
    Regularize an overlap matrix to ensure positive-definiteness.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

__all__ = [
    "solve_generalized_eigenvalue",
    "compute_ground_state_energy",
    "analyze_spectrum",
    "regularize_overlap_matrix",
]

logger = logging.getLogger(__name__)

# Type alias for matrices accepted by the solvers
_MatrixLike = Union[np.ndarray, scipy.sparse.spmatrix]


# ---------------------------------------------------------------------------
# Generalized eigenvalue solver
# ---------------------------------------------------------------------------


def solve_generalized_eigenvalue(
    H: _MatrixLike,
    S: _MatrixLike,
    k: int = 1,
    which: str = "SA",
    use_gpu: bool = False,
    davidson_threshold: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigenvalue problem Hv = ESv.

    Dispatches to the appropriate backend depending on matrix format,
    size, and whether GPU acceleration is requested.

    Solver selection (from highest to lowest priority):

    1. **GPU** — if *use_gpu* is ``True`` and CuPy is available.
    2. **Sparse** — if *H* or *S* is a SciPy sparse matrix.
    3. **Davidson** — if the matrix dimension ``n`` exceeds
       *davidson_threshold* (iterative, much faster than dense for
       large systems when only a few eigenvalues are needed).
    4. **Dense** — full eigendecomposition via ``scipy.linalg.eigh``.

    Parameters
    ----------
    H : np.ndarray or scipy.sparse.spmatrix
        Hamiltonian matrix, shape ``(n, n)``.
    S : np.ndarray or scipy.sparse.spmatrix
        Overlap matrix, shape ``(n, n)``.  Must be positive-definite;
        use :func:`regularize_overlap_matrix` if necessary.
    k : int, optional
        Number of lowest eigenvalues / eigenvectors to compute
        (default ``1``).
    which : str, optional
        Eigenvalue selection criterion: ``"SA"`` for smallest algebraic
        (default).
    use_gpu : bool, optional
        If ``True``, attempt to use CuPy for GPU-accelerated computation.
        Falls back to CPU if CuPy is unavailable.
    davidson_threshold : int, optional
        Minimum matrix dimension above which the Davidson iterative
        solver is preferred over full dense decomposition (default
        ``500``).  Ignored when the matrix is sparse.

    Returns
    -------
    eigenvalues : np.ndarray
        The lowest ``k`` eigenvalues, shape ``(k,)``, sorted ascending.
    eigenvectors : np.ndarray
        Corresponding eigenvectors, shape ``(n, k)``.

    Raises
    ------
    ValueError
        If ``H`` and ``S`` have incompatible shapes or ``k < 1``.
    RuntimeError
        If the eigensolver fails to converge.

    Examples
    --------
    >>> H = np.diag([1.0, 2.0, 3.0])
    >>> S = np.eye(3)
    >>> vals, vecs = solve_generalized_eigenvalue(H, S, k=2)
    >>> vals
    array([1., 2.])
    """
    if H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got shape {H.shape}")
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square, got shape {S.shape}")
    if H.shape != S.shape:
        raise ValueError(
            f"H and S must have the same shape, got {H.shape} and {S.shape}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    n = H.shape[0]

    if use_gpu:
        result = _solve_gpu(H, S, k, which)
        if result is not None:
            return result
        logger.info("CuPy unavailable; falling back to CPU eigensolver.")

    is_sparse = scipy.sparse.issparse(H) or scipy.sparse.issparse(S)

    if is_sparse and k < n - 1:
        return _solve_sparse(H, S, k, which)

    # Davidson only supports smallest algebraic eigenvalues
    if n > davidson_threshold and k < n and which == "SA":
        return _solve_davidson(H, S, k)

    return _solve_dense(H, S, k)


def _solve_dense(
    H: _MatrixLike,
    S: _MatrixLike,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve with scipy.linalg.eigh (dense, full spectrum then truncate).

    Parameters
    ----------
    H : array-like
        Hamiltonian matrix.
    S : array-like
        Overlap matrix.
    k : int
        Number of eigenvalues to return.

    Returns
    -------
    eigenvalues : np.ndarray
        Shape ``(k,)``.
    eigenvectors : np.ndarray
        Shape ``(n, k)``.
    """
    H_dense = H.toarray() if scipy.sparse.issparse(H) else np.asarray(H)
    S_dense = S.toarray() if scipy.sparse.issparse(S) else np.asarray(S)

    eigenvalues, eigenvectors = scipy.linalg.eigh(H_dense, S_dense)
    order = np.argsort(eigenvalues)
    return eigenvalues[order[:k]], eigenvectors[:, order[:k]]


def _solve_davidson(
    H: _MatrixLike,
    S: _MatrixLike,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve using the Davidson iterative eigensolver.

    For the generalized case (S != I) the problem is transformed to
    standard form via Cholesky factorization of S before applying
    Davidson, then the eigenvectors are back-transformed.

    Parameters
    ----------
    H : array-like
        Hamiltonian matrix.
    S : array-like
        Overlap matrix.
    k : int
        Number of eigenvalues to return.

    Returns
    -------
    eigenvalues : np.ndarray
        Shape ``(k,)``.
    eigenvectors : np.ndarray
        Shape ``(n, k)``.
    """
    from qvartools.diag.eigen.davidson import DavidsonSolver

    H_dense = (
        H.toarray() if scipy.sparse.issparse(H) else np.asarray(H, dtype=np.float64)
    )
    S_dense = (
        S.toarray() if scipy.sparse.issparse(S) else np.asarray(S, dtype=np.float64)
    )

    # Check if S is identity (standard eigenvalue problem)
    # Avoid allocating a full n×n identity matrix for the comparison
    n = S_dense.shape[0]
    diag_ok = np.allclose(np.diag(S_dense), 1.0, atol=1e-12)
    if diag_ok:
        # Check off-diagonal: sum of abs values minus trace
        off_diag_norm = np.abs(S_dense).sum() - np.abs(np.diag(S_dense)).sum()
        is_identity = off_diag_norm < n * 1e-12
    else:
        is_identity = False

    if is_identity:
        H_work = H_dense
        L = None
    else:
        # Transform to standard form: L^{-1} H L^{-T} where S = L L^T
        # Use solve_triangular to avoid forming explicit L^{-1} (saves O(n^2) memory)
        try:
            L = scipy.linalg.cholesky(S_dense, lower=True)
            # Step 1: temp = L^{-1} H  (solve L @ temp = H)
            temp = scipy.linalg.solve_triangular(L, H_dense, lower=True)
            # Step 2: H_work = temp @ L^{-T} = (L^{-1} temp^T)^T
            # Since H is symmetric: H_work = L^{-1} H L^{-T} is also symmetric
            H_work = scipy.linalg.solve_triangular(L, temp.T, lower=True).T
        except scipy.linalg.LinAlgError:
            logger.warning("Cholesky failed for S; falling back to dense eigh.")
            return _solve_dense(H, S, k)

    solver = DavidsonSolver(max_iterations=200, tolerance=1e-8)
    try:
        eigenvalues, eigenvectors = solver.solve(H_work, k=k)
    except (RuntimeError, ValueError):
        logger.warning("Davidson solver failed; falling back to dense eigh.")
        return _solve_dense(H, S, k)

    if L is not None:
        # Back-transform: v_original = L^{-T} v_standard
        eigenvectors = scipy.linalg.solve_triangular(L.T, eigenvectors, lower=False)

    logger.debug("Davidson solver used for n=%d, k=%d", H_dense.shape[0], k)
    return eigenvalues, eigenvectors


def _solve_sparse(
    H: _MatrixLike,
    S: _MatrixLike,
    k: int,
    which: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve with scipy.sparse.linalg.eigsh (sparse, iterative).

    Parameters
    ----------
    H : sparse matrix
        Hamiltonian.
    S : sparse matrix
        Overlap.
    k : int
        Number of eigenvalues.
    which : str
        Selection criterion.

    Returns
    -------
    eigenvalues : np.ndarray
        Shape ``(k,)``.
    eigenvectors : np.ndarray
        Shape ``(n, k)``.
    """
    H_sp = scipy.sparse.csr_matrix(H) if not scipy.sparse.issparse(H) else H
    S_sp = scipy.sparse.csr_matrix(S) if not scipy.sparse.issparse(S) else S

    try:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            H_sp, k=k, M=S_sp, which=which
        )
    except scipy.sparse.linalg.ArpackNoConvergence as exc:
        raise RuntimeError(
            f"Sparse eigensolver did not converge after maximum iterations. "
            f"Converged values: {len(exc.eigenvalues)}/{k}"
        ) from exc

    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def _solve_gpu(
    H: _MatrixLike,
    S: _MatrixLike,
    k: int,
    which: str = "SA",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Attempt GPU solve via CuPy.

    Parameters
    ----------
    H : array-like
        Hamiltonian matrix.
    S : array-like
        Overlap matrix.
    k : int
        Number of eigenvalues.
    which : str, optional
        Eigenvalue selection criterion (default ``"SA"``).

    Returns
    -------
    tuple of np.ndarray or None
        ``(eigenvalues, eigenvectors)`` if CuPy is available, else ``None``.
    """
    try:
        import cupy as cp  # type: ignore[import-untyped]
        import cupyx.scipy.linalg as cupy_linalg  # type: ignore[import-untyped]
    except ImportError:
        return None

    is_sparse = scipy.sparse.issparse(H) or scipy.sparse.issparse(S)

    if is_sparse:
        # Use CuPy sparse eigsh to avoid densifying large matrices
        try:
            import cupyx.scipy.sparse as cusp_sparse  # type: ignore[import-untyped]
            import cupyx.scipy.sparse.linalg as cusp_linalg  # type: ignore[import-untyped]

            H_sp = scipy.sparse.csr_matrix(H) if not scipy.sparse.issparse(H) else H
            S_sp = scipy.sparse.csr_matrix(S) if not scipy.sparse.issparse(S) else S

            H_gpu = cusp_sparse.csr_matrix(H_sp)
            S_gpu = cusp_sparse.csr_matrix(S_sp)

            eigenvalues_gpu, eigenvectors_gpu = cusp_linalg.eigsh(
                H_gpu, k=k, M=S_gpu, which=which
            )
            eigenvalues = cp.asnumpy(eigenvalues_gpu)
            eigenvectors = cp.asnumpy(eigenvectors_gpu)

            order = np.argsort(eigenvalues)
            logger.debug("GPU sparse eigsh: k=%d", k)
            return eigenvalues[order[:k]], eigenvectors[:, order[:k]]
        except Exception as exc:
            logger.warning(
                "CuPy sparse eigsh failed (%s); falling back to CPU sparse.",
                exc,
            )
            return None

    # Dense GPU path (only reached for dense inputs)
    H_dense = H.toarray() if scipy.sparse.issparse(H) else np.asarray(H)
    S_dense = S.toarray() if scipy.sparse.issparse(S) else np.asarray(S)

    H_gpu = cp.asarray(H_dense)
    S_gpu = cp.asarray(S_dense)

    eigenvalues_gpu, eigenvectors_gpu = cupy_linalg.eigh(H_gpu, S_gpu)

    eigenvalues = cp.asnumpy(eigenvalues_gpu)
    eigenvectors = cp.asnumpy(eigenvectors_gpu)

    order = np.argsort(eigenvalues)
    return eigenvalues[order[:k]], eigenvectors[:, order[:k]]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def compute_ground_state_energy(
    H: _MatrixLike,
    use_gpu: bool = False,
) -> float:
    """Compute the ground-state energy (lowest eigenvalue) of a Hamiltonian.

    Parameters
    ----------
    H : np.ndarray or scipy.sparse.spmatrix
        Hamiltonian matrix, shape ``(n, n)``.
    use_gpu : bool, optional
        If ``True``, attempt GPU acceleration via CuPy.

    Returns
    -------
    float
        The ground-state energy.

    Examples
    --------
    >>> H = np.diag([3.0, 1.0, 2.0])
    >>> compute_ground_state_energy(H)
    1.0
    """
    S = scipy.sparse.eye(H.shape[0], format="csr")
    eigenvalues, _ = solve_generalized_eigenvalue(H, S, k=1, use_gpu=use_gpu)
    return float(eigenvalues[0])


def analyze_spectrum(
    H: _MatrixLike,
    k: int = 6,
    use_gpu: bool = False,
) -> dict:
    """Compute spectral statistics of a Hamiltonian.

    Parameters
    ----------
    H : np.ndarray or scipy.sparse.spmatrix
        Hamiltonian matrix, shape ``(n, n)``.
    k : int, optional
        Number of lowest eigenvalues to compute (default ``6``).
    use_gpu : bool, optional
        If ``True``, attempt GPU acceleration.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"eigenvalues"`` : np.ndarray -- the lowest ``k`` eigenvalues.
        - ``"gaps"`` : np.ndarray -- consecutive eigenvalue gaps, shape
          ``(k-1,)``.
        - ``"first_excited_gap"`` : float -- gap between ground and first
          excited state.
        - ``"ground_state_energy"`` : float -- the lowest eigenvalue.

    Examples
    --------
    >>> H = np.diag([1.0, 3.0, 7.0])
    >>> info = analyze_spectrum(H, k=3)
    >>> info["first_excited_gap"]
    2.0
    """
    n = H.shape[0]
    actual_k = min(k, n)

    S = scipy.sparse.eye(n, format="csr")
    eigenvalues, _ = solve_generalized_eigenvalue(H, S, k=actual_k, use_gpu=use_gpu)

    gaps = np.diff(eigenvalues)
    first_gap = float(gaps[0]) if len(gaps) > 0 else 0.0

    return {
        "eigenvalues": eigenvalues,
        "gaps": gaps,
        "first_excited_gap": first_gap,
        "ground_state_energy": float(eigenvalues[0]),
    }


# ---------------------------------------------------------------------------
# Overlap matrix regularization
# ---------------------------------------------------------------------------


def regularize_overlap_matrix(
    S: _MatrixLike,
    threshold: float = 1e-6,
    use_gpu: bool = False,
) -> scipy.sparse.csr_matrix:
    """Regularize an overlap matrix to ensure positive-definiteness.

    Diagonalizes ``S``, clamps eigenvalues below ``threshold`` to
    ``threshold``, and reconstructs a well-conditioned overlap matrix.

    Parameters
    ----------
    S : np.ndarray or scipy.sparse.spmatrix
        Overlap matrix, shape ``(n, n)``.
    threshold : float, optional
        Minimum allowed eigenvalue (default ``1e-6``).
    use_gpu : bool, optional
        If ``True``, attempt GPU acceleration via CuPy.

    Returns
    -------
    scipy.sparse.csr_matrix
        Regularized overlap matrix in CSR format.

    Examples
    --------
    >>> S = np.array([[1.0, 0.99], [0.99, 1.0]])
    >>> S_reg = regularize_overlap_matrix(S, threshold=0.1)
    """
    S_dense = (
        S.toarray() if scipy.sparse.issparse(S) else np.asarray(S, dtype=np.float64)
    )

    if use_gpu:
        try:
            import cupy as cp  # type: ignore[import-untyped]

            S_gpu = cp.asarray(S_dense)
            eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(S_gpu)
            eigenvalues = cp.asnumpy(eigenvalues_gpu)
            eigenvectors = cp.asnumpy(eigenvectors_gpu)
        except ImportError:
            logger.info("CuPy unavailable; falling back to CPU for regularization.")
            eigenvalues, eigenvectors = np.linalg.eigh(S_dense)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(S_dense)

    n_clamped = int(np.sum(eigenvalues < threshold))
    if n_clamped > 0:
        logger.info(
            "Clamping %d eigenvalues below threshold %.2e to %.2e",
            n_clamped,
            threshold,
            threshold,
        )

    eigenvalues_clamped = np.maximum(eigenvalues, threshold)

    S_regularized = eigenvectors @ np.diag(eigenvalues_clamped) @ eigenvectors.T

    # Symmetrize to remove floating-point asymmetry
    S_regularized = 0.5 * (S_regularized + S_regularized.T)

    return scipy.sparse.csr_matrix(S_regularized)
