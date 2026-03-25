"""Tests for eigensolvers.

Sets KMP_DUPLICATE_LIB_OK to work around OpenMP conflicts between torch
and scipy on Windows.
"""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest

from qvartools.diag.eigen.eigensolver import (
    DavidsonSolver,
    compute_ground_state_energy,
    regularize_overlap_matrix,
    solve_generalized_eigenvalue,
)


class TestSolveGeneralizedEigenvalue:
    """Tests for solve_generalized_eigenvalue."""

    def test_known_2x2(self) -> None:
        H = np.array([[2.0, 0.0], [0.0, 5.0]])
        S = np.eye(2)
        vals, vecs = solve_generalized_eigenvalue(H, S, k=2)
        np.testing.assert_allclose(vals, [2.0, 5.0], atol=1e-10)

    def test_returns_k_eigenvalues(self) -> None:
        H = np.diag([1.0, 3.0, 5.0, 7.0])
        S = np.eye(4)
        vals, vecs = solve_generalized_eigenvalue(H, S, k=2)
        assert vals.shape == (2,)
        assert vecs.shape == (4, 2)
        np.testing.assert_allclose(vals, [1.0, 3.0], atol=1e-10)

    def test_invalid_k_raises(self) -> None:
        H = np.eye(3)
        S = np.eye(3)
        with pytest.raises(ValueError, match="k must be >= 1"):
            solve_generalized_eigenvalue(H, S, k=0)

    def test_shape_mismatch_raises(self) -> None:
        H = np.eye(3)
        S = np.eye(4)
        with pytest.raises(ValueError, match="same shape"):
            solve_generalized_eigenvalue(H, S, k=1)


class TestComputeGroundStateEnergy:
    """Tests for compute_ground_state_energy."""

    def test_diagonal_matrix(self) -> None:
        H = np.diag([3.0, 1.0, 2.0])
        energy = compute_ground_state_energy(H)
        assert abs(energy - 1.0) < 1e-10

    def test_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        H = A + A.T  # symmetric
        energy = compute_ground_state_energy(H)
        numpy_vals = np.linalg.eigvalsh(H)
        assert abs(energy - numpy_vals[0]) < 1e-8


class TestRegularizeOverlapMatrix:
    """Tests for regularize_overlap_matrix."""

    def test_produces_positive_definite(self) -> None:
        # Construct a nearly singular overlap matrix
        S = np.array([[1.0, 0.999], [0.999, 1.0]])
        S_reg = regularize_overlap_matrix(S, threshold=0.01)
        S_dense = S_reg.toarray()

        eigenvalues = np.linalg.eigvalsh(S_dense)
        assert (eigenvalues >= 0.01 - 1e-10).all()

    def test_already_positive_definite_unchanged(self) -> None:
        S = np.eye(3)
        S_reg = regularize_overlap_matrix(S, threshold=1e-6)
        S_dense = S_reg.toarray()
        np.testing.assert_allclose(S_dense, np.eye(3), atol=1e-10)

    def test_symmetric_output(self) -> None:
        rng = np.random.default_rng(123)
        A = rng.standard_normal((4, 4))
        S = A @ A.T + 0.01 * np.eye(4)
        S_reg = regularize_overlap_matrix(S)
        S_dense = S_reg.toarray()
        np.testing.assert_allclose(S_dense, S_dense.T, atol=1e-12)


class TestDavidsonSolver:
    """Tests for DavidsonSolver."""

    def test_diagonal_matrix(self) -> None:
        H = np.diag([5.0, 1.0, 3.0, 2.0, 4.0])
        solver = DavidsonSolver(max_iterations=100, tolerance=1e-10)
        vals, vecs = solver.solve(H, k=2)
        np.testing.assert_allclose(vals, [1.0, 2.0], atol=1e-6)

    def test_symmetric_matrix(self) -> None:
        rng = np.random.default_rng(99)
        A = rng.standard_normal((8, 8))
        H = A + A.T
        solver = DavidsonSolver(max_iterations=200, tolerance=1e-8)
        vals, vecs = solver.solve(H, k=2)

        numpy_vals = np.sort(np.linalg.eigvalsh(H))
        np.testing.assert_allclose(vals, numpy_vals[:2], atol=1e-5)

    def test_k_exceeds_dimension_raises(self) -> None:
        H = np.eye(3)
        solver = DavidsonSolver()
        with pytest.raises(ValueError, match="exceeds matrix dimension"):
            solver.solve(H, k=5)
