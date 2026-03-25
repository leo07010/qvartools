"""
_utils --- Internal utilities for qvartools
============================================

Provides caching, automatic parameter scaling, GPU-accelerated linear
algebra, format conversion, and basis expansion utilities used across
the qvartools pipeline.

Re-exported symbols
-------------------
ConnectionCache
    Hash-based cache for Hamiltonian connections.
SystemScaler
    Automatic parameter scaling based on system size.
SystemMetrics
    Diagnostic metrics for a quantum system.
ScaledParameters
    Auto-scaled pipeline hyperparameters.
QualityPreset
    Quality preset enum (FAST, BALANCED, ACCURATE).
SystemTier
    System classification by Hilbert-space size.
gpu_solve_fermion
    GPU-accelerated FCI energy computation.
configs_to_ibm_format
    Convert binary configs to IBM SQD bitstring format.
ibm_format_to_configs
    Convert IBM SQD format back to binary configs.
vectorized_dedup
    Efficient set-difference deduplication of configurations.
expand_basis_via_connections
    Basis expansion via Hamiltonian connections.
ConfigHash
    Type alias for overflow-safe configuration hashes.
config_integer_hash
    Batch integer hashing for binary configurations.
compute_occupancies
    Orbital occupancies from an eigenvector.
"""

from qvartools._utils.hashing.config_hash import ConfigHash, config_integer_hash
from qvartools._utils.hashing.connection_cache import ConnectionCache
from qvartools._utils.formatting.bitstring_format import (
    configs_to_ibm_format,
    ibm_format_to_configs,
    vectorized_dedup,
)
from qvartools._utils.gpu.diagnostics import compute_occupancies
from qvartools._utils.gpu.diagnostics import gpu_solve_fermion as gpu_solve_fermion_diag
from qvartools._utils.gpu.linear_algebra import gpu_solve_fermion
from qvartools.krylov.expansion.krylov_expand import expand_basis_via_connections
from qvartools._utils.scaling.quality_presets import (
    QualityPreset,
    ScaledParameters,
    SystemMetrics,
    SystemTier,
)
from qvartools._utils.scaling.system_scaler import SystemScaler

__all__ = [
    "ConnectionCache",
    "SystemScaler",
    "SystemMetrics",
    "ScaledParameters",
    "QualityPreset",
    "SystemTier",
    "gpu_solve_fermion",
    "gpu_solve_fermion_diag",
    "configs_to_ibm_format",
    "ibm_format_to_configs",
    "vectorized_dedup",
    "expand_basis_via_connections",
    "ConfigHash",
    "config_integer_hash",
    "compute_occupancies",
]
