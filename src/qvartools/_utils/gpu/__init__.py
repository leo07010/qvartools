"""gpu --- GPU-accelerated linear algebra and diagonalisation."""
from __future__ import annotations

from qvartools._utils.gpu.linear_algebra import gpu_solve_fermion
from qvartools._utils.gpu.diagnostics import compute_occupancies
from qvartools._utils.gpu.diagnostics import gpu_solve_fermion as gpu_solve_fermion_diag
from qvartools._utils.gpu.fci_solver import (
    GPU_FCI_AVAILABLE,
    GPUFCISolver,
    compute_gpu_fci,
    compute_gpu_fci_from_integrals,
)

__all__ = [
    "gpu_solve_fermion",
    "gpu_solve_fermion_diag",
    "compute_occupancies",
    "GPU_FCI_AVAILABLE",
    "GPUFCISolver",
    "compute_gpu_fci",
    "compute_gpu_fci_from_integrals",
]
