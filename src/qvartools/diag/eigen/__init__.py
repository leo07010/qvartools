"""eigen --- Eigensolvers and projected-Hamiltonian construction."""
from __future__ import annotations

from qvartools.diag.eigen.davidson import DavidsonSolver
from qvartools.diag.eigen.eigenvalue import (
    compute_ground_state_energy,
    regularize_overlap_matrix,
    solve_generalized_eigenvalue,
)
from qvartools.diag.eigen.projected_hamiltonian import (
    ProjectedHamiltonianBuilder,
    ProjectedHamiltonianConfig,
)

__all__ = [
    "DavidsonSolver",
    "compute_ground_state_energy",
    "regularize_overlap_matrix",
    "solve_generalized_eigenvalue",
    "ProjectedHamiltonianBuilder",
    "ProjectedHamiltonianConfig",
]
