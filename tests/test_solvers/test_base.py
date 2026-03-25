"""Tests for solver base classes."""

from __future__ import annotations

import pytest

from qvartools.solvers.solver import Solver, SolverResult


# ---------------------------------------------------------------------------
# SolverResult
# ---------------------------------------------------------------------------


class TestSolverResult:
    """Tests for the SolverResult frozen dataclass."""

    def test_fields(self):
        result = SolverResult(
            energy=-1.5,
            diag_dim=100,
            wall_time=2.5,
            method="FCI",
            converged=True,
            metadata={"iterations": 10},
        )
        assert result.energy == -1.5
        assert result.diag_dim == 100
        assert result.wall_time == 2.5
        assert result.method == "FCI"
        assert result.converged is True
        assert result.metadata == {"iterations": 10}

    def test_default_metadata(self):
        result = SolverResult(
            energy=-1.0, diag_dim=50, wall_time=1.0, method="SQD", converged=False
        )
        assert result.metadata == {}

    def test_immutable(self):
        result = SolverResult(
            energy=-1.0, diag_dim=50, wall_time=1.0, method="SQD", converged=True
        )
        with pytest.raises(AttributeError):
            result.energy = -2.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            result.method = "FCI"  # type: ignore[misc]

    def test_repr(self):
        result = SolverResult(
            energy=-1.5, diag_dim=100, wall_time=2.5, method="FCI", converged=True
        )
        repr_str = repr(result)
        assert "FCI" in repr_str
        assert "-1.5" in repr_str


# ---------------------------------------------------------------------------
# Solver ABC
# ---------------------------------------------------------------------------


class TestSolverABC:
    """Tests for the Solver abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Solver()  # type: ignore[abstract]

    def test_subclass_must_implement_solve(self):
        class IncompleteSolver(Solver):
            pass

        with pytest.raises(TypeError):
            IncompleteSolver()  # type: ignore[abstract]

    def test_subclass_with_solve(self):
        class DummySolver(Solver):
            def solve(self, hamiltonian, mol_info):
                return SolverResult(
                    energy=-1.0,
                    diag_dim=10,
                    wall_time=0.1,
                    method="Dummy",
                    converged=True,
                )

        solver = DummySolver()
        result = solver.solve(None, {})  # type: ignore[arg-type]
        assert result.method == "Dummy"
