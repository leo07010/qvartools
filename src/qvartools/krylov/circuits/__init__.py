"""circuits --- Quantum-circuit Krylov methods."""
from __future__ import annotations

from qvartools.krylov.circuits.spectral import compute_optimal_dt
from qvartools.krylov.circuits.circuit_skqd import (
    QuantumSKQDConfig,
    QuantumCircuitSKQD,
)
from qvartools.krylov.circuits.sqd import (
    SQDConfig,
    SQDSolver,
    inject_depolarizing_noise,
)

__all__: list[str] = [
    "compute_optimal_dt",
    "QuantumSKQDConfig",
    "QuantumCircuitSKQD",
    "SQDConfig",
    "SQDSolver",
    "inject_depolarizing_noise",
]
