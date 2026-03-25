"""
eigensolver --- Backward-compatible re-exports
===============================================

This module re-exports symbols that were split into
:mod:`~qvartools.diag.eigenvalue` and
:mod:`~qvartools.diag.davidson` for backward compatibility.
"""

from qvartools.diag.eigen.davidson import DavidsonSolver
from qvartools.diag.eigen.eigenvalue import (
    analyze_spectrum,
    compute_ground_state_energy,
    regularize_overlap_matrix,
    solve_generalized_eigenvalue,
)

__all__ = [
    "solve_generalized_eigenvalue",
    "compute_ground_state_energy",
    "analyze_spectrum",
    "regularize_overlap_matrix",
    "DavidsonSolver",
]
