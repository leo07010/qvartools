"""
qvartools — Quantum Variational Toolkit
========================================

A unified Python package for quantum variational methods applied to
molecular ground-state energy estimation. Consolidates normalizing-flow
guided neural quantum states (NF-NQS), sample-based quantum
diagonalization (SQD), and sample-based Krylov quantum diagonalization
(SKQD) into reusable, well-scoped modules.

Subpackages
-----------
hamiltonians
    Hamiltonian representations (molecular, spin) with efficient matrix
    element computation.
nqs
    Neural quantum state architectures (dense, complex, RBM, transformer).
flows
    Normalizing flows for configuration sampling, including
    particle-conserving variants and physics-guided training.
krylov
    Krylov subspace methods: SKQD, residual expansion, basis sampling,
    and quantum-circuit variants.
diag
    Subspace diagonalization: eigensolvers, diversity selection,
    projected Hamiltonian construction.
solvers
    High-level solver interfaces (FCI, CCSD, SQD, SKQD, iterative).
samplers
    Configuration samplers (NF, Trotter, CUDA-Q).
molecules
    Molecular system registry and integral computation.
methods
    End-to-end method pipelines (HI-NQS-SQD, HI-NQS-SKQD, NQS-SQD,
    NQS-SKQD).
_utils
    Internal utilities (caching, GPU helpers, format conversion).
"""

__version__ = "0.0.0"

from qvartools._logging import configure_logging, get_logger
from qvartools.pipeline import (
    FlowGuidedKrylovPipeline,
    PipelineConfig,
    run_molecular_benchmark,
)

# Auto-configure logging from environment on import
configure_logging()

__all__ = [
    "hamiltonians",
    "nqs",
    "flows",
    "krylov",
    "diag",
    "solvers",
    "samplers",
    "molecules",
    "methods",
    "PipelineConfig",
    "FlowGuidedKrylovPipeline",
    "run_molecular_benchmark",
    "configure_logging",
    "get_logger",
]
