"""
krylov --- Krylov subspace methods for quantum diagonalization
==============================================================

This subpackage implements sample-based Krylov quantum diagonalization
(SKQD) and supporting algorithms for iterative basis expansion and
configuration sampling.

Classes
-------
SKQDConfig
    Hyperparameters for the SKQD algorithm.
SampleBasedKrylovDiagonalization
    Core SKQD solver: constructs Krylov states, samples configurations,
    and solves the projected eigenvalue problem.
FlowGuidedSKQD
    SKQD variant seeded with normalizing-flow basis states for
    accelerated convergence.
ResidualExpansionConfig
    Hyperparameters for residual-based and selected-CI expansion.
ResidualBasedExpander
    Iterative basis expansion driven by residual analysis.
SelectedCIExpander
    CIPSI-style selected-CI basis expansion using perturbative importance.
KrylovBasisSampler
    Interface for sampling configurations from Krylov-evolved states.
"""

from qvartools.krylov.basis.sampler import KrylovBasisSampler
from qvartools.krylov.basis.flow_guided import FlowGuidedSKQD
from qvartools.krylov.expansion.residual_config import ResidualExpansionConfig
from qvartools.krylov.expansion.residual_expander import ResidualBasedExpander
from qvartools.krylov.expansion.selected_ci_expander import SelectedCIExpander
from qvartools.krylov.basis.skqd import SKQDConfig, SampleBasedKrylovDiagonalization

__all__ = [
    "SKQDConfig",
    "SampleBasedKrylovDiagonalization",
    "FlowGuidedSKQD",
    "ResidualExpansionConfig",
    "ResidualBasedExpander",
    "SelectedCIExpander",
    "KrylovBasisSampler",
]
