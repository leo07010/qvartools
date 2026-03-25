"""
samplers --- Configuration samplers for quantum diagonalization
===============================================================

This subpackage provides configuration samplers that generate
computational-basis states for use in sample-based quantum
diagonalization workflows.

Classes
-------
SamplerResult
    Immutable dataclass holding sampled configurations and metadata.
Sampler
    Abstract base class defining the sampler interface.
NFSampler
    Normalizing-flow-based sampler wrapping a trained flow model.
TrotterSampler
    Classical Trotterized time-evolution sampler.
CUDAQCircuitSampler
    CUDA-Q UCCSD circuit sampler (requires ``cudaq``).
LUCJSampler
    LUCJ circuit sampler using Qiskit + ffsim (requires ``qiskit``,
    ``ffsim``).
TransformerNFSampler
    Transformer-based normalizing-flow sampler.
"""

from qvartools.samplers.sampler import Sampler, SamplerResult
from qvartools.samplers.classical.nf_sampler import NFSampler
from qvartools.samplers.quantum.trotter_sampler import TrotterSampler
from qvartools.samplers.classical.transformer_nf_sampler import (
    TransformerNFSampler,
    TransformerSamplerConfig,
)

__all__ = [
    "Sampler",
    "SamplerResult",
    "NFSampler",
    "TrotterSampler",
    "TransformerNFSampler",
    "TransformerSamplerConfig",
]


# Optional-dependency samplers: import lazily to avoid hard failures
def __getattr__(name: str):
    if name == "CUDAQCircuitSampler":
        from qvartools.samplers.quantum.cudaq_sampler import CUDAQCircuitSampler

        return CUDAQCircuitSampler
    if name == "CUDAQSamplerConfig":
        from qvartools.samplers.quantum.cudaq_sampler import CUDAQSamplerConfig

        return CUDAQSamplerConfig
    if name == "LUCJSampler":
        from qvartools.samplers.quantum.lucj_sampler import LUCJSampler

        return LUCJSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
