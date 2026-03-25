"""
solvers --- High-level solver interfaces for quantum chemistry
==============================================================

This subpackage provides a unified solver interface for computing
ground-state energies of molecular Hamiltonians.  All solvers inherit
from :class:`Solver` and return a :class:`SolverResult` dataclass.

Classes
-------
SolverResult
    Immutable dataclass holding energy, timing, and convergence metadata.
Solver
    Abstract base class defining the solver interface.
FCISolver
    Full configuration interaction solver (PySCF or dense fallback).
CCSDSolver
    Coupled cluster singles and doubles solver via PySCF.
SQDSolver
    Sample-based quantum diagonalization with normalizing-flow sampling.
SQDBatchedSolver
    SQD with self-consistent batching and orbital occupancy updates.
SKQDSolver
    Sample-based Krylov quantum diagonalization with normalizing-flow
    sampling and residual expansion.
NFSKQDSolver
    NF-SKQD: faithful NF analog of quantum SKQD with cumulative basis.
SKQDSolverB
    SKQD Method B: H-connection expansion with PT2 importance scoring.
SKQDSolverC
    SKQD Method C: H-connection expansion with coupling strength scoring.
CIPSISolver
    CIPSI selected-CI solver with perturbative determinant selection.
DCISKQDSolverB
    DCI-SKQD Method B: CIPSI seed + PT2 importance expansion.
DCISKQDSolverC
    DCI-SKQD Method C: CIPSI seed + coupling strength expansion.
IterativeNFSQDSolver
    Hardware-informed iterative NF-SQD with eigenvector feedback.
IterativeNFSKQDSolver
    Hardware-informed iterative NF-SKQD with eigenvector feedback.
"""

from qvartools.solvers.solver import Solver, SolverResult
from qvartools.solvers.reference.ccsd import CCSDSolver
from qvartools.solvers.subspace.cipsi import CIPSISolver
from qvartools.solvers.krylov.dci_skqd import DCISKQDSolverB, DCISKQDSolverC
from qvartools.solvers.reference.fci import FCISolver
from qvartools.solvers.iterative.iterative_skqd import IterativeNFSKQDSolver
from qvartools.solvers.iterative.iterative_sqd import IterativeNFSQDSolver
from qvartools.solvers.krylov.nf_skqd import NFSKQDSolver
from qvartools.solvers.krylov.skqd import SKQDSolver
from qvartools.solvers.krylov.skqd_expansion import SKQDSolverB, SKQDSolverC
from qvartools.solvers.subspace.sqd import SQDSolver
from qvartools.solvers.subspace.sqd_batched import SQDBatchedSolver

__all__ = [
    "SolverResult",
    "Solver",
    "FCISolver",
    "CCSDSolver",
    "SQDSolver",
    "SQDBatchedSolver",
    "SKQDSolver",
    "NFSKQDSolver",
    "SKQDSolverB",
    "SKQDSolverC",
    "CIPSISolver",
    "DCISKQDSolverB",
    "DCISKQDSolverC",
    "IterativeNFSQDSolver",
    "IterativeNFSKQDSolver",
]
