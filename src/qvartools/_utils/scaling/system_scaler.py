"""
system_scaler --- Automatic parameter scaling based on system size
=================================================================

Provides :class:`SystemScaler`, which analyses a Hamiltonian's Hilbert-space
size and automatically computes well-tuned pipeline hyperparameters.

Classes
-------
SystemScaler
    Stateless scaler that converts Hamiltonian metadata into parameters.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict

from qvartools._utils.scaling.quality_presets import (
    QualityPreset,
    ScaledParameters,
    SystemMetrics,
)

__all__ = [
    "SystemScaler",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preset multipliers
# ---------------------------------------------------------------------------

_PRESET_MULTIPLIERS: Dict[QualityPreset, Dict[str, float]] = {
    QualityPreset.FAST: {
        "hidden": 0.6,
        "samples": 0.5,
        "epochs": 0.5,
        "basis": 0.5,
        "krylov": 0.7,
        "lr": 1.5,
    },
    QualityPreset.BALANCED: {
        "hidden": 1.0,
        "samples": 1.0,
        "epochs": 1.0,
        "basis": 1.0,
        "krylov": 1.0,
        "lr": 1.0,
    },
    QualityPreset.ACCURATE: {
        "hidden": 1.5,
        "samples": 2.0,
        "epochs": 1.5,
        "basis": 2.0,
        "krylov": 1.5,
        "lr": 0.7,
    },
}


# ---------------------------------------------------------------------------
# SystemScaler
# ---------------------------------------------------------------------------


class SystemScaler:
    """Automatic parameter scaler for quantum variational pipelines.

    Analyses a Hamiltonian's Hilbert-space size and produces a complete
    :class:`ScaledParameters` instance with well-tuned hyperparameters.

    Parameters
    ----------
    preset : QualityPreset, optional
        Quality preset controlling the accuracy--speed trade-off
        (default :attr:`QualityPreset.BALANCED`).

    Examples
    --------
    >>> from qvartools._utils.scaling.system_scaler import SystemScaler
    >>> from qvartools._utils.scaling.quality_presets import QualityPreset
    >>> scaler = SystemScaler(preset=QualityPreset.FAST)
    >>> config = scaler.create_pipeline_config(hamiltonian)
    """

    def __init__(
        self, preset: QualityPreset = QualityPreset.BALANCED
    ) -> None:
        self.preset: QualityPreset = preset
        self._mult: Dict[str, float] = _PRESET_MULTIPLIERS[preset]

    # ------------------------------------------------------------------
    # System analysis
    # ------------------------------------------------------------------

    def analyze_system(self, hamiltonian: Any) -> SystemMetrics:
        """Analyse a Hamiltonian and return system-size metrics.

        For molecular Hamiltonians with ``integrals`` attribute, the number
        of valid configurations is computed combinatorially from the
        electron and orbital counts.  For generic Hamiltonians the full
        Hilbert-space dimension is used.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            A Hamiltonian instance (molecular or spin).

        Returns
        -------
        SystemMetrics
            Diagnostic metrics for the system.
        """
        n_qubits = hamiltonian.num_sites

        # Molecular Hamiltonian: count particle-conserving configurations
        if hasattr(hamiltonian, "integrals"):
            integrals = hamiltonian.integrals
            n_orb = integrals.n_orbitals
            n_alpha = integrals.n_alpha
            n_beta = integrals.n_beta
            n_valid = math.comb(n_orb, n_alpha) * math.comb(n_orb, n_beta)
        else:
            # Generic Hamiltonian: use full Hilbert-space dimension
            n_valid = hamiltonian.hilbert_dim

        logger.info(
            "System analysis: n_qubits=%d, n_valid_configs=%d",
            n_qubits,
            n_valid,
        )

        return SystemMetrics(n_qubits=n_qubits, n_valid_configs=n_valid)

    # ------------------------------------------------------------------
    # Parameter computation
    # ------------------------------------------------------------------

    def compute_parameters(self, metrics: SystemMetrics) -> ScaledParameters:
        """Compute scaled pipeline hyperparameters from system metrics.

        Scaling formulas
        ~~~~~~~~~~~~~~~~
        * ``hidden_dim ~ log2(n_configs) * 16``
        * ``samples ~ sqrt(n_configs) * 32`` (capped at 100 000)
        * ``epochs ~ log10(n_configs) * 200`` (capped at 5 000)
        * ``krylov_dim ~ log2(n_configs) / 2`` (capped at 30)

        Parameters
        ----------
        metrics : SystemMetrics
            System-size metrics from :meth:`analyze_system`.

        Returns
        -------
        ScaledParameters
            Complete set of auto-scaled hyperparameters.
        """
        m = self._mult
        n = max(metrics.n_valid_configs, 2)

        log2_n = math.log2(n)
        log10_n = math.log10(n)
        sqrt_n = math.sqrt(n)

        # --- Network architecture ---
        base_hidden = int(log2_n * 16 * m["hidden"])
        base_hidden = max(base_hidden, 32)
        # Two hidden layers: wider first, narrower second
        hidden_dims = (base_hidden, base_hidden // 2)

        # --- Sampling ---
        samples_per_batch = int(sqrt_n * 32 * m["samples"])
        samples_per_batch = max(64, min(samples_per_batch, 100_000))

        num_batches = max(1, int(log10_n * 2 * m["samples"]))
        num_batches = min(num_batches, 50)

        # --- Training ---
        max_epochs = int(log10_n * 200 * m["epochs"])
        max_epochs = max(50, min(max_epochs, 5_000))

        min_epochs = max(10, max_epochs // 5)

        convergence_threshold = 1e-6 / m["epochs"]

        flow_lr = 1e-3 * m["lr"]
        nqs_lr = 5e-4 * m["lr"]

        # --- Loss weights ---
        teacher_weight = 1.0
        physics_weight = min(0.5 + log10_n * 0.1, 2.0)
        entropy_weight = max(0.01, 0.1 / log10_n)

        # --- Basis management ---
        max_accumulated_basis = int(sqrt_n * 10 * m["basis"])
        max_accumulated_basis = max(100, min(max_accumulated_basis, 50_000))

        max_diverse_configs = int(sqrt_n * 5 * m["basis"])
        max_diverse_configs = max(50, min(max_diverse_configs, 20_000))

        rank_2_fraction = 0.3

        # --- Residual expansion ---
        residual_iterations = max(3, int(log10_n * 2 * m["basis"]))
        residual_iterations = min(residual_iterations, 20)

        residual_configs_per_iter = int(sqrt_n * 2 * m["basis"])
        residual_configs_per_iter = max(20, min(residual_configs_per_iter, 5_000))

        residual_threshold = 1e-4 / m["basis"]

        # --- Krylov ---
        max_krylov_dim = int(log2_n / 2 * m["krylov"])
        max_krylov_dim = max(3, min(max_krylov_dim, 30))

        time_step = 0.1
        shots_per_krylov = int(sqrt_n * 100 * m["samples"])
        shots_per_krylov = max(1_000, min(shots_per_krylov, 1_000_000))

        skqd_regularization = 1e-6

        return ScaledParameters(
            hidden_dims=hidden_dims,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            convergence_threshold=convergence_threshold,
            flow_lr=flow_lr,
            nqs_lr=nqs_lr,
            teacher_weight=teacher_weight,
            physics_weight=physics_weight,
            entropy_weight=entropy_weight,
            max_accumulated_basis=max_accumulated_basis,
            max_diverse_configs=max_diverse_configs,
            rank_2_fraction=rank_2_fraction,
            residual_iterations=residual_iterations,
            residual_configs_per_iter=residual_configs_per_iter,
            residual_threshold=residual_threshold,
            max_krylov_dim=max_krylov_dim,
            time_step=time_step,
            shots_per_krylov=shots_per_krylov,
            skqd_regularization=skqd_regularization,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def create_pipeline_config(self, hamiltonian: Any) -> Dict[str, Any]:
        """Analyse a Hamiltonian and return a pipeline configuration dict.

        This is a convenience method that chains :meth:`analyze_system` and
        :meth:`compute_parameters`, returning the result as a plain
        dictionary suitable for passing to pipeline constructors.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            A Hamiltonian instance.

        Returns
        -------
        dict
            Dictionary with all pipeline hyperparameters plus ``"metrics"``
            and ``"tier"`` keys for diagnostics.
        """
        metrics = self.analyze_system(hamiltonian)
        params = self.compute_parameters(metrics)

        config: Dict[str, Any] = {
            # Diagnostics
            "metrics": {
                "n_qubits": metrics.n_qubits,
                "n_valid_configs": metrics.n_valid_configs,
                "tier": metrics.tier.value,
                "log_configs": metrics.log_configs,
                "log10_configs": metrics.log10_configs,
                "sqrt_configs": metrics.sqrt_configs,
            },
            "tier": metrics.tier.value,
            # Network
            "hidden_dims": list(params.hidden_dims),
            # Sampling
            "samples_per_batch": params.samples_per_batch,
            "num_batches": params.num_batches,
            # Training
            "max_epochs": params.max_epochs,
            "min_epochs": params.min_epochs,
            "convergence_threshold": params.convergence_threshold,
            "flow_lr": params.flow_lr,
            "nqs_lr": params.nqs_lr,
            # Loss weights
            "teacher_weight": params.teacher_weight,
            "physics_weight": params.physics_weight,
            "entropy_weight": params.entropy_weight,
            # Basis
            "max_accumulated_basis": params.max_accumulated_basis,
            "max_diverse_configs": params.max_diverse_configs,
            "rank_2_fraction": params.rank_2_fraction,
            # Residual expansion
            "residual_iterations": params.residual_iterations,
            "residual_configs_per_iter": params.residual_configs_per_iter,
            "residual_threshold": params.residual_threshold,
            # Krylov
            "max_krylov_dim": params.max_krylov_dim,
            "time_step": params.time_step,
            "shots_per_krylov": params.shots_per_krylov,
            "skqd_regularization": params.skqd_regularization,
        }

        logger.info(
            "Pipeline config for %s system (n_qubits=%d, n_configs=%d): "
            "hidden=%s, samples=%d, epochs=%d, krylov=%d",
            metrics.tier.value,
            metrics.n_qubits,
            metrics.n_valid_configs,
            list(params.hidden_dims),
            params.samples_per_batch,
            params.max_epochs,
            params.max_krylov_dim,
        )

        return config
