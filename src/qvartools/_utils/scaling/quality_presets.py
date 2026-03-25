"""
quality_presets --- Enumerations and dataclasses for system scaling
===================================================================

Provides the enumerations and dataclasses used by
:class:`~qvartools._utils.system_scaler.SystemScaler` to classify systems
and represent auto-scaled pipeline hyperparameters.

Classes
-------
QualityPreset
    Enum of ``FAST``, ``BALANCED``, ``ACCURATE``.
SystemTier
    Enum classifying systems by Hilbert-space size.
SystemMetrics
    Dataclass holding system-size diagnostics.
ScaledParameters
    Dataclass holding all auto-scaled pipeline hyperparameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "QualityPreset",
    "SystemTier",
    "SystemMetrics",
    "ScaledParameters",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class QualityPreset(Enum):
    """Quality preset controlling the accuracy--speed trade-off.

    Attributes
    ----------
    FAST
        Prioritises speed; fewer samples, shorter training, smaller basis.
    BALANCED
        Default trade-off; reasonable accuracy with moderate compute.
    ACCURATE
        Prioritises accuracy; more samples, longer training, larger basis.
    """

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class SystemTier(Enum):
    """Classification of systems by the number of valid configurations.

    Attributes
    ----------
    TINY
        Fewer than 1 000 configurations.
    SMALL
        1 000 -- 10 000 configurations.
    MEDIUM
        10 000 -- 100 000 configurations.
    LARGE
        100 000 -- 1 000 000 configurations.
    VERY_LARGE
        1 000 000 -- 10 000 000 configurations.
    HUGE
        More than 10 000 000 configurations.
    """

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"
    HUGE = "huge"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemMetrics:
    """Diagnostic metrics derived from a Hamiltonian's Hilbert space.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (sites) in the system.
    n_valid_configs : int
        Number of physically valid configurations (e.g. particle-number
        conserving states).

    Properties
    ----------
    tier : SystemTier
        Classification of the system by configuration count.
    log_configs : float
        Natural logarithm of ``n_valid_configs``.
    log10_configs : float
        Base-10 logarithm of ``n_valid_configs``.
    sqrt_configs : float
        Square root of ``n_valid_configs``.
    """

    n_qubits: int
    n_valid_configs: int

    @property
    def tier(self) -> SystemTier:
        """Classify the system by number of valid configurations.

        Returns
        -------
        SystemTier
        """
        n = self.n_valid_configs
        if n < 1_000:
            return SystemTier.TINY
        if n < 10_000:
            return SystemTier.SMALL
        if n < 100_000:
            return SystemTier.MEDIUM
        if n < 1_000_000:
            return SystemTier.LARGE
        if n < 10_000_000:
            return SystemTier.VERY_LARGE
        return SystemTier.HUGE

    @property
    def log_configs(self) -> float:
        """Natural logarithm of ``n_valid_configs``.

        Returns
        -------
        float
        """
        return math.log(max(self.n_valid_configs, 1))

    @property
    def log10_configs(self) -> float:
        """Base-10 logarithm of ``n_valid_configs``.

        Returns
        -------
        float
        """
        return math.log10(max(self.n_valid_configs, 1))

    @property
    def sqrt_configs(self) -> float:
        """Square root of ``n_valid_configs``.

        Returns
        -------
        float
        """
        return math.sqrt(self.n_valid_configs)


@dataclass(frozen=True)
class ScaledParameters:
    """Auto-scaled pipeline hyperparameters.

    All fields are computed by :meth:`SystemScaler.compute_parameters`
    based on the system size and the chosen quality preset.

    Parameters
    ----------
    hidden_dims : list of int
        Hidden-layer widths for neural-network architectures.
    samples_per_batch : int
        Number of configuration samples per batch.
    num_batches : int
        Number of batches per epoch.
    max_epochs : int
        Maximum number of training epochs.
    min_epochs : int
        Minimum number of training epochs before early stopping.
    convergence_threshold : float
        Energy convergence threshold for early stopping (Hartree).
    flow_lr : float
        Learning rate for the normalizing-flow optimiser.
    nqs_lr : float
        Learning rate for the NQS optimiser.
    teacher_weight : float
        Weight for the teacher (flow) loss term.
    physics_weight : float
        Weight for the physics (energy) loss term.
    entropy_weight : float
        Weight for the entropy regularisation term.
    max_accumulated_basis : int
        Maximum total basis size across all iterations.
    max_diverse_configs : int
        Maximum number of diverse configurations to select.
    rank_2_fraction : float
        Fraction of basis allocated to rank-2 (double-excitation) configs.
    residual_iterations : int
        Number of residual-expansion iterations.
    residual_configs_per_iter : int
        Configurations added per residual-expansion iteration.
    residual_threshold : float
        Minimum residual magnitude for inclusion.
    max_krylov_dim : int
        Maximum Krylov subspace dimension.
    time_step : float
        Imaginary-time step for Krylov propagation.
    shots_per_krylov : int
        Number of shots per Krylov vector measurement.
    skqd_regularization : float
        Tikhonov regularisation for the SKQD generalised eigenvalue problem.
    """

    hidden_dims: tuple  # frozen dataclass needs hashable type
    samples_per_batch: int
    num_batches: int
    max_epochs: int
    min_epochs: int
    convergence_threshold: float
    flow_lr: float
    nqs_lr: float
    teacher_weight: float
    physics_weight: float
    entropy_weight: float
    max_accumulated_basis: int
    max_diverse_configs: int
    rank_2_fraction: float
    residual_iterations: int
    residual_configs_per_iter: int
    residual_threshold: float
    max_krylov_dim: int
    time_step: float
    shots_per_krylov: int
    skqd_regularization: float
