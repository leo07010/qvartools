"""
residual_expansion --- Backward-compatibility shim
===================================================

This module re-exports all public symbols that were originally defined here
before the decomposition into :mod:`residual_config`,
:mod:`residual_expander`, and :mod:`selected_ci_expander`.

Existing imports such as
``from qvartools.krylov.expansion.residual_expansion import _diagonalise_in_basis``
continue to work unchanged.
"""

from qvartools.krylov.expansion.residual_config import (
    ResidualExpansionConfig,
    _diagonalise_in_basis,
    _generate_candidate_configs,
)
from qvartools.krylov.expansion.residual_expander import ResidualBasedExpander
from qvartools.krylov.expansion.selected_ci_expander import SelectedCIExpander

__all__ = [
    "ResidualExpansionConfig",
    "ResidualBasedExpander",
    "SelectedCIExpander",
    "_diagonalise_in_basis",
    "_generate_candidate_configs",
]
