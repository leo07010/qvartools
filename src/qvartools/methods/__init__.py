"""
methods --- High-level method pipelines
========================================

Complete method wrappers that compose samplers, solvers, and NQS models
into end-to-end energy-estimation pipelines.  Each runner function
accepts a configuration dataclass and returns a :class:`SolverResult`.

Functions
---------
run_hi_nqs_sqd
    HI+NQS+SQD: iterative NQS-SQD self-consistent loop.
run_hi_nqs_skqd
    HI+NQS+SKQD: NQS sampling + Krylov expansion + GPU diag.
run_nqs_skqd
    NQS+SKQD: two-stage (train NQS then SKQD).
run_nqs_sqd
    NQS+SQD: two-stage (train NQS then SQD).
"""

from __future__ import annotations

from qvartools.methods.nqs.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
from qvartools.methods.nqs.hi_nqs_skqd import HINQSSKQDConfig, run_hi_nqs_skqd
from qvartools.methods.nqs.nqs_skqd import NQSSKQDConfig, run_nqs_skqd
from qvartools.methods.nqs.nqs_sqd import NQSSQDConfig, run_nqs_sqd

__all__ = [
    "HINQSSQDConfig",
    "run_hi_nqs_sqd",
    "HINQSSKQDConfig",
    "run_hi_nqs_skqd",
    "NQSSKQDConfig",
    "run_nqs_skqd",
    "NQSSQDConfig",
    "run_nqs_sqd",
]
