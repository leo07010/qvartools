"""scaling --- Quality presets and system-size auto-scaling."""
from __future__ import annotations

from qvartools._utils.scaling.quality_presets import (
    QualityPreset,
    ScaledParameters,
    SystemMetrics,
    SystemTier,
)
from qvartools._utils.scaling.system_scaler import SystemScaler

__all__ = [
    "QualityPreset",
    "ScaledParameters",
    "SystemMetrics",
    "SystemTier",
    "SystemScaler",
]
