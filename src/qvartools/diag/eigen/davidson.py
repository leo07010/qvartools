"""
davidson --- Iterative Davidson eigensolver
============================================

Provides the :class:`DavidsonSolver` iterative eigensolver targeting the
lowest eigenvalues of large sparse Hermitian matrices.

Classes
-------
DavidsonSolver
    Iterative Davidson eigensolver targeting the lowest eigenvalues of
    large sparse Hermitian matrices.
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse

__all__ = [
    "DavidsonSolver",
]

logger = logging.getLogger(__name__)

# Type alias for matrices accepted by the solver
_MatrixLike = Union[np.ndarray, scipy.sparse.spmatrix]


class DavidsonSolver:
    """Iterative Davidson eigensolver for large sparse Hermitian matrices.

    The Davidson algorithm extends the power method by building a subspace
    of approximate eigenvectors and using a preconditioner (the diagonal of
    the matrix) to accelerate convergence.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of Davidson iterations (default ``100``).
    tolerance : float, optional
        Convergence tolerance on the residual norm (default ``1e-8``).
    max_subspace_size : int, optional
        Maximum subspace dimension before restart (default ``20``).

    Examples
    --------
    >>> from scipy.sparse import random as sparse_random
    >>> H = sparse_random(100, 100, density=0.1, format="csr")
    >>> H = H + H.T  # symmetrize
    >>> solver = DavidsonSolver(max_iterations=200)
    >>> eigenvalues, eigenvectors = solver.solve(H, k=2)
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        max_subspace_size: int = 20,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")
        if max_subspace_size < 2:
            raise ValueError(f"max_subspace_size must be >= 2, got {max_subspace_size}")

        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._max_subspace_size = max_subspace_size

    def solve(
        self,
        H: _MatrixLike,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the lowest ``k`` eigenvalues and eigenvectors of ``H``.

        Parameters
        ----------
        H : np.ndarray or scipy.sparse.spmatrix
            Hermitian matrix, shape ``(n, n)``.
        k : int, optional
            Number of lowest eigenpairs to compute (default ``1``).

        Returns
        -------
        eigenvalues : np.ndarray
            Lowest ``k`` eigenvalues, shape ``(k,)``, sorted ascending.
        eigenvectors : np.ndarray
            Corresponding eigenvectors, shape ``(n, k)``.

        Raises
        ------
        ValueError
            If ``k`` exceeds the matrix dimension.
        RuntimeError
            If the solver does not converge within ``max_iterations``.
        """
        n = H.shape[0]
        if k > n:
            raise ValueError(
                f"k={k} exceeds matrix dimension n={n}"
            )

        H_sparse = scipy.sparse.csr_matrix(H) if not scipy.sparse.issparse(H) else H
        diag = np.array(H_sparse.diagonal(), dtype=np.float64)

        # Initialize subspace with k unit vectors corresponding to the
        # smallest diagonal elements (better starting point for Davidson)
        diag_order = np.argsort(diag)
        V = np.zeros((n, k), dtype=np.float64)
        for j in range(k):
            V[diag_order[j], j] = 1.0

        converged = False
        eigenvalues = np.zeros(k)
        eigenvectors = np.zeros((n, k))
        max_residual = float("inf")

        for iteration in range(self._max_iterations):
            # Orthonormalize subspace
            V, _ = np.linalg.qr(V, mode="reduced")

            # Project H into subspace
            HV = H_sparse @ V
            H_proj = V.T @ HV

            # Solve small projected problem
            theta, s = scipy.linalg.eigh(H_proj)
            # Keep only the lowest k
            theta = theta[:k]
            s = s[:, :k]

            # Ritz vectors
            ritz = V @ s

            # Compute residuals: r_j = H @ ritz_j - theta_j * ritz_j
            H_ritz = H_sparse @ ritz
            residuals = H_ritz - ritz * theta[np.newaxis, :]
            max_residual = 0.0
            for j in range(k):
                res_norm = float(np.linalg.norm(residuals[:, j]))
                max_residual = max(max_residual, res_norm)

            logger.debug(
                "Davidson iteration %d: max residual = %.2e",
                iteration,
                max_residual,
            )

            if max_residual < self._tolerance:
                eigenvalues = theta
                eigenvectors = ritz
                converged = True
                logger.info(
                    "Davidson converged in %d iterations (residual=%.2e)",
                    iteration + 1,
                    max_residual,
                )
                break

            # Expand subspace with preconditioned residuals
            new_vectors = []
            for j in range(k):
                r = residuals[:, j]
                # Diagonal preconditioner: (D - theta_j * I)^{-1} r
                precond_diag = diag - theta[j]
                # Avoid division by zero / near-zero
                precond_diag = np.where(
                    np.abs(precond_diag) < 1e-12,
                    np.sign(precond_diag + 1e-30) * 1e-12,
                    precond_diag,
                )
                t = -r / precond_diag
                # Orthogonalize against current subspace via modified
                # Gram-Schmidt (two passes for numerical stability)
                for _pass in range(2):
                    t = t - V @ (V.T @ t)
                t_norm = np.linalg.norm(t)
                if t_norm > 1e-14:
                    new_vectors.append(t / t_norm)

            if len(new_vectors) == 0:
                eigenvalues = theta
                eigenvectors = ritz
                converged = True
                break

            V_new = np.column_stack(new_vectors)
            V = np.column_stack([V, V_new])

            # Restart if subspace grows too large
            if V.shape[1] > self._max_subspace_size:
                V = ritz.copy()

        if not converged:
            raise RuntimeError(
                f"Davidson solver did not converge after {self._max_iterations} "
                f"iterations (residual={max_residual:.2e}, tol={self._tolerance:.2e})"
            )

        sort_order = np.argsort(eigenvalues)
        return eigenvalues[sort_order], eigenvectors[:, sort_order]
