"""quantum --- Quantum-circuit configuration samplers."""
from __future__ import annotations

from qvartools.samplers.quantum.trotter_sampler import TrotterSampler

__all__ = ["TrotterSampler"]

# Optional imports guarded by try/except
def __getattr__(name: str):
    if name in ("CUDAQCircuitSampler", "CUDAQSamplerConfig"):
        from qvartools.samplers.quantum.cudaq_sampler import CUDAQCircuitSampler, CUDAQSamplerConfig
        return {"CUDAQCircuitSampler": CUDAQCircuitSampler, "CUDAQSamplerConfig": CUDAQSamplerConfig}[name]
    if name == "LUCJSampler":
        from qvartools.samplers.quantum.lucj_sampler import LUCJSampler
        return LUCJSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
