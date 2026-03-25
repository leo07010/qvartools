"""expansion --- Basis expansion strategies (residual, selected-CI, connections)."""
from __future__ import annotations

from qvartools.krylov.expansion.residual_config import ResidualExpansionConfig
from qvartools.krylov.expansion.residual_expander import ResidualBasedExpander
from qvartools.krylov.expansion.selected_ci_expander import SelectedCIExpander
from qvartools.krylov.expansion.krylov_expand import expand_basis_via_connections

__all__ = [
    "ResidualExpansionConfig",
    "ResidualBasedExpander",
    "SelectedCIExpander",
    "expand_basis_via_connections",
]
