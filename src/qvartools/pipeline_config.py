"""
pipeline_config --- Hyperparameter configuration for the pipeline
=================================================================

Holds :class:`PipelineConfig`, a dataclass with every hyperparameter
needed by :class:`~qvartools.pipeline.FlowGuidedKrylovPipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

__all__ = [
    "PipelineConfig",
]


@dataclass
class PipelineConfig:
    """Hyperparameters for the flow-guided Krylov / SQD pipeline.

    Supports three subspace diagonalization modes:

    - ``"classical_krylov"``: Classical exact time evolution (no Trotter error).
    - ``"skqd"``: Real SKQD via quantum circuit Trotterized evolution (CUDA-Q).
    - ``"sqd"``: IBM SQD sampling-based batch diagonalization.

    When ``skip_nf_training`` is ``True``, the pipeline operates in
    **Direct-CI mode**: it generates HF + singles + doubles without NF
    training, then proceeds directly to subspace diagonalization.

    Parameters
    ----------
    use_particle_conserving_flow : bool
        Use a particle-conserving flow architecture (default ``True``).
    nf_hidden_dims : list of int
        Hidden-layer dimensions for the normalizing flow.
    nqs_hidden_dims : list of int
        Hidden-layer dimensions for the NQS.
    samples_per_batch : int
        Training samples drawn per batch.
    num_batches : int
        Number of batches per training epoch.
    max_epochs : int
        Maximum training epochs.
    min_epochs : int
        Minimum training epochs before convergence check.
    convergence_threshold : float
        Relative energy change for convergence.
    teacher_weight : float
        Weight of the teacher (KL) loss term.
    physics_weight : float
        Weight of the physics (energy) loss term.
    entropy_weight : float
        Weight of the entropy regularization term.
    flow_lr : float
        Learning rate for the normalizing flow.
    nqs_lr : float
        Learning rate for the NQS.
    max_accumulated_basis : int
        Hard limit on accumulated basis size.
    use_diversity_selection : bool
        Apply diversity-aware selection to the basis.
    max_diverse_configs : int
        Maximum configurations after diversity selection.
    rank_2_fraction : float
        Fraction of the diversity budget for double excitations.
    use_residual_expansion : bool
        Enable residual / perturbative basis expansion.
    residual_iterations : int
        Number of residual expansion iterations.
    residual_configs_per_iter : int
        Configurations added per residual iteration.
    residual_threshold : float
        Minimum residual magnitude for inclusion.
    use_perturbative_selection : bool
        Use CIPSI-style perturbative selection instead of residual.
    subspace_mode : str
        Subspace diag backend: ``"classical_krylov"`` (default, exact time
        evolution), ``"skqd"`` (CUDA-Q Trotterized circuits),
        ``"skqd_quantum"`` (alias for ``"skqd"``), or ``"sqd"``.
    sqd_num_batches : int
        Number of random batches for SQD.
    sqd_batch_size : int
        Configurations per SQD batch (``0`` = auto).
    sqd_self_consistent_iters : int
        Self-consistent occupancy iterations for SQD.
    sqd_spin_penalty : float
        Spin-penalty coefficient for SQD.
    sqd_noise_rate : float
        Noise rate for SQD-Recovery mode (``0`` = clean).
    sqd_use_spin_symmetry : bool
        Enable spin-symmetry enhancement in SQD.
    max_krylov_dim : int
        Maximum Krylov dimension for SKQD.
    time_step : float
        Time step for SKQD evolution.
    shots_per_krylov : int
        Measurement shots per Krylov state.
    skqd_regularization : float
        Tikhonov regularization for the SKQD overlap matrix.
    skip_skqd : bool
        Skip SKQD and use direct diagonalization.
    auto_time_step : bool
        Auto-compute time step from spectral range.
    quantum_num_trotter_steps : int
        Trotter steps for quantum SKQD.
    quantum_total_evolution_time : float
        Total evolution time for quantum SKQD.
    quantum_shots : int
        Shots for quantum circuit measurements.
    quantum_cudaq_target : str
        CUDA-Q target backend.
    quantum_cudaq_option : str
        CUDA-Q precision option.
    use_local_energy : bool
        Use local-energy estimator during training.
    use_ci_seeding : bool
        Seed flow training with CI configurations.
    use_davidson : bool
        Use Davidson eigensolver for large matrices.
    davidson_threshold : int
        Basis size above which Davidson is preferred.
    skip_nf_training : bool
        Skip NF training (Direct-CI mode).
    device : str
        Torch device string.
    max_connections_per_config : int
        Max Hamiltonian connections per config (``0`` = unlimited).
    diagonal_only_warmup_epochs : int
        Epochs using diagonal-only Hamiltonian at start.
    stochastic_connections_fraction : float
        Fraction of connections to sample stochastically.

    Examples
    --------
    >>> cfg = PipelineConfig(max_epochs=200, subspace_mode="sqd")
    >>> cfg.subspace_mode
    'sqd'
    """

    # --- Flow type ---
    use_particle_conserving_flow: bool = True

    # --- NQS model selection ---
    nqs_type: str = "dense"  # "dense", "signed", "complex", "rbm", "transformer"

    # --- NF-NQS architecture ---
    nf_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    nqs_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256, 256])

    # --- Training parameters ---
    samples_per_batch: int = 2000
    num_batches: int = 1
    max_epochs: int = 400
    min_epochs: int = 100
    convergence_threshold: float = 0.20
    teacher_weight: float = 1.0
    physics_weight: float = 0.0
    entropy_weight: float = 0.0
    flow_lr: float = 5e-4
    nqs_lr: float = 1e-3

    # --- Basis management ---
    max_accumulated_basis: int = 4096
    use_diversity_selection: bool = True
    max_diverse_configs: int = 2048
    rank_2_fraction: float = 0.50

    # --- Residual / perturbative expansion ---
    use_residual_expansion: bool = True
    residual_iterations: int = 8
    residual_configs_per_iter: int = 150
    residual_threshold: float = 1e-6
    use_perturbative_selection: bool = True

    # --- Subspace diagonalization mode ---
    subspace_mode: str = "classical_krylov"  # "classical_krylov", "skqd", "sqd"

    # --- SQD-specific parameters ---
    sqd_num_batches: int = 5
    sqd_batch_size: int = 0  # 0 = auto from NF samples
    sqd_self_consistent_iters: int = 3
    sqd_spin_penalty: float = 0.0
    sqd_noise_rate: float = 0.0  # 0 = clean SQD; >0 = SQD-Recovery mode
    sqd_use_spin_symmetry: bool = True

    # --- SKQD parameters ---
    max_krylov_dim: int = 15
    time_step: float = 0.1
    shots_per_krylov: int = 100_000
    skqd_regularization: float = 1e-8
    skip_skqd: bool = False
    auto_time_step: bool = True  # compute dt = pi / spectral_range

    # --- Quantum circuit SKQD parameters (subspace_mode="skqd_quantum") ---
    quantum_num_trotter_steps: int = 1
    quantum_total_evolution_time: float = 3.14159
    quantum_shots: int = 100_000
    quantum_cudaq_target: str = "nvidia"
    quantum_cudaq_option: str = "fp64"

    # --- Training mode ---
    use_local_energy: bool = True
    use_ci_seeding: bool = False

    # --- Eigensolver ---
    use_davidson: bool = True
    davidson_threshold: int = 500

    # --- Direct-CI mode ---
    skip_nf_training: bool = False

    # --- Hardware ---
    device: str = "cpu"

    # --- Performance optimizations for large systems ---
    max_connections_per_config: int = 0
    diagonal_only_warmup_epochs: int = 0
    stochastic_connections_fraction: float = 1.0

    def adapt_to_system_size(
        self,
        n_valid_configs: int,
        verbose: bool = True,
    ) -> PipelineConfig:
        """Return a new config with parameters scaled for the given system size.

        Classifies the Hilbert-space size into four tiers and adjusts
        **only** the basis limits and NQS network dimensions.  Training
        hyperparameters (samples, epochs, batches, learning rates) and
        SKQD parameters (krylov_dim, shots) are **preserved** at their
        paper-aligned defaults for small/medium systems to ensure
        accuracy.

        The adaptation strategy follows the original Flow-Guided-Krylov
        branches: only basis capacity, NQS capacity, and (for very
        large systems) epoch/sample budgets are adjusted.

        Parameters
        ----------
        n_valid_configs : int
            Number of valid (particle-conserving) configurations in the
            Hilbert space.  For spin systems, use ``2**num_sites``.
        verbose : bool
            If ``True``, print adaptation diagnostics.

        Returns
        -------
        PipelineConfig
            A new config with scaled hyperparameters.

        Examples
        --------
        >>> cfg = PipelineConfig()
        >>> adapted = cfg.adapt_to_system_size(500, verbose=False)
        >>> adapted.max_diverse_configs <= 500
        True
        """
        if n_valid_configs <= 1_000:
            tier = "small"
        elif n_valid_configs <= 5_000:
            tier = "medium"
        elif n_valid_configs <= 20_000:
            tier = "large"
        else:
            tier = "very_large"

        if verbose:
            print(f"System size: {n_valid_configs:,} valid configs -> {tier} tier")

        # Small / medium: only adjust basis limits, keep all other defaults
        if tier == "small":
            result = replace(
                self,
                max_accumulated_basis=max(n_valid_configs, 4096),
                max_diverse_configs=min(n_valid_configs, 2048),
            )
        elif tier == "medium":
            result = replace(
                self,
                nqs_hidden_dims=[384, 384, 384, 384, 384],
                max_accumulated_basis=min(n_valid_configs, 8192),
                max_diverse_configs=min(n_valid_configs, 4096),
            )
        elif tier == "large":
            result = replace(
                self,
                nqs_hidden_dims=[512, 512, 512, 512, 512],
                samples_per_batch=4000,
                max_epochs=max(self.max_epochs, 600),
                max_accumulated_basis=min(n_valid_configs, 12288),
                max_diverse_configs=min(n_valid_configs, 8192),
            )
        else:  # very_large
            result = replace(
                self,
                nqs_hidden_dims=[512, 512, 512, 512],
                samples_per_batch=2000,
                max_epochs=max(self.max_epochs, 200),
                min_epochs=max(self.min_epochs, 50),
                max_accumulated_basis=16384,
                max_diverse_configs=min(n_valid_configs, 12288),
                max_krylov_dim=4,
            )

        if verbose:
            print("Adapted parameters:")
            print(f"  subspace_mode: {result.subspace_mode}")
            print(f"  max_accumulated_basis: {result.max_accumulated_basis:,}")
            print(f"  max_diverse_configs: {result.max_diverse_configs:,}")

        return result
