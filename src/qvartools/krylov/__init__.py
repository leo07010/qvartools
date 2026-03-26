"""
krylov --- Krylov subspace methods for quantum diagonalization
==============================================================

This subpackage implements classical Krylov diagonalization and
supporting algorithms for iterative basis expansion and configuration
sampling.

Classes
-------
SKQDConfig
    Hyperparameters for the Krylov diagonalization algorithm.
ClassicalKrylovDiagonalization
    Core classical Krylov solver: constructs Krylov states via exact
    matrix exponentiation, samples configurations, and solves the
    projected eigenvalue problem.
FlowGuidedKrylovDiag
    Classical Krylov variant seeded with normalizing-flow basis states
    for accelerated convergence.
ResidualExpansionConfig
    Hyperparameters for residual-based and selected-CI expansion.
ResidualBasedExpander
    Iterative basis expansion driven by residual analysis.
SelectedCIExpander
    CIPSI-style selected-CI basis expansion using perturbative importance.
KrylovBasisSampler
    Interface for sampling configurations from Krylov-evolved states.
"""

from qvartools.krylov.basis.flow_guided import FlowGuidedKrylovDiag
from qvartools.krylov.basis.sampler import KrylovBasisSampler
from qvartools.krylov.basis.skqd import (
    ClassicalKrylovDiagonalization,
    SKQDConfig,
)
from qvartools.krylov.expansion.residual_config import ResidualExpansionConfig
from qvartools.krylov.expansion.residual_expander import ResidualBasedExpander
from qvartools.krylov.expansion.selected_ci_expander import SelectedCIExpander

__all__ = [
    "SKQDConfig",
    "ClassicalKrylovDiagonalization",
    "FlowGuidedKrylovDiag",
    "ResidualExpansionConfig",
    "ResidualBasedExpander",
    "SelectedCIExpander",
    "KrylovBasisSampler",
    # Deprecated aliases (remove in v0.1.0) — resolved via __getattr__
    "SampleBasedKrylovDiagonalization",
    "FlowGuidedSKQD",
]


def __getattr__(name: str):
    """Emit DeprecationWarning for deprecated aliases."""
    import warnings

    _DEPRECATED = {
        "SampleBasedKrylovDiagonalization": (
            "ClassicalKrylovDiagonalization",
            ClassicalKrylovDiagonalization,
        ),
        "FlowGuidedSKQD": (
            "FlowGuidedKrylovDiag",
            FlowGuidedKrylovDiag,
        ),
    }
    if name in _DEPRECATED:
        new_name, cls = _DEPRECATED[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead. "
            "The old name will be removed in v0.1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
