"""
diag --- Subspace diagonalization and basis selection
=====================================================

Core eigensolvers, diversity-aware basis selection, and projected
Hamiltonian construction for sample-based quantum diagonalization.

Classes
-------
DiversityConfig
    Hyperparameters controlling diversity-aware configuration selection.
DiversitySelector
    Selects a diverse, representative subset of configurations from a
    sampled pool using excitation-rank bucketing and Hamming-distance
    filtering.
DavidsonSolver
    Iterative Davidson eigensolver for large sparse Hamiltonians.
ProjectedHamiltonianConfig
    Hyperparameters for projected Hamiltonian construction.
ProjectedHamiltonianBuilder
    Constructs the projected Hamiltonian matrix in a sampled basis.

Functions
---------
solve_generalized_eigenvalue
    Solve the generalized eigenvalue problem Hv = ESv.
compute_ground_state_energy
    Compute the ground-state energy from a Hamiltonian matrix.
regularize_overlap_matrix
    Regularize an overlap matrix for positive-definiteness.
"""

from qvartools.diag.eigen.davidson import DavidsonSolver
from qvartools.diag.selection.diversity_selection import (
    DiversityConfig,
    DiversitySelector,
)
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
    "DiversityConfig",
    "DiversitySelector",
    "solve_generalized_eigenvalue",
    "compute_ground_state_energy",
    "regularize_overlap_matrix",
    "DavidsonSolver",
    "ProjectedHamiltonianConfig",
    "ProjectedHamiltonianBuilder",
]
