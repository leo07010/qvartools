"""solvers --- Reference solvers (FCI, CCSD, CIPSI/SCI)."""

from __future__ import annotations

from qvartools.solvers.solver import Solver, SolverResult
from qvartools.solvers.reference.fci import FCISolver
from qvartools.solvers.reference.ccsd import CCSDSolver, CCSDTSolver
from qvartools.solvers.subspace.cipsi import CIPSISolver

__all__ = [
    "Solver",
    "SolverResult",
    "FCISolver",
    "CCSDSolver",
    "CCSDTSolver",
    "CIPSISolver",
]
