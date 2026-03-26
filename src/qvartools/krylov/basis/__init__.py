"""basis --- Krylov basis construction and diagonalization drivers."""

from __future__ import annotations

from qvartools.krylov.basis.flow_guided import FlowGuidedKrylovDiag
from qvartools.krylov.basis.sampler import KrylovBasisSampler
from qvartools.krylov.basis.skqd import (
    ClassicalKrylovDiagonalization,
    SKQDConfig,
)

__all__ = [
    "KrylovBasisSampler",
    "SKQDConfig",
    "ClassicalKrylovDiagonalization",
    "FlowGuidedKrylovDiag",
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
