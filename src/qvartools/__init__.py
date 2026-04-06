"""
qvartools — HI+NQS+SQD v3
==========================

Self-consistent neural quantum state + IBM SQD ground-state solver.

Subpackages
-----------
hamiltonians
    Molecular Hamiltonian with Slater-Condon matrix elements (PySCF).
nqs
    Autoregressive Transformer NQS (Psiformer-style).
solvers
    Reference solvers: FCI, CCSD, CIPSI/SCI.
molecules
    Molecular system registry (H2O, NH3, N2, C2H2, C2H4, CAS systems).
methods
    HI+NQS+SQD v3 end-to-end pipeline.
"""

__version__ = "0.1.0"

from qvartools._logging import configure_logging, get_logger
from qvartools.methods.nqs import run_hi_nqs_sqd, HINQSSQDConfig
from qvartools.molecules import get_molecule

configure_logging()

__all__ = [
    "run_hi_nqs_sqd",
    "HINQSSQDConfig",
    "get_molecule",
    "configure_logging",
    "get_logger",
]
