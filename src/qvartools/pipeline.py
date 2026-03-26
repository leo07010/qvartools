"""
pipeline --- Top-level orchestrator for flow-guided Krylov diagonalization
==========================================================================

Ties together all qvartools subpackages into a single, configurable pipeline
that executes the flow-guided SKQD / SQD workflow:

1. **Train** a normalizing flow + neural quantum state via physics-guided
   mixed-objective training (or generate Direct-CI basis if skipped).
2. **Extract and select** a diverse, representative basis set from the
   accumulated flow samples.
3. **Diagonalize** in the selected subspace via SKQD (Krylov),
   SKQD-Quantum (Trotterized circuit), or SQD (batch diag).

Classes
-------
FlowGuidedKrylovPipeline
    Main orchestrator that wires stages 1--3 together.

Functions
---------
run_molecular_benchmark
    Convenience function to load a molecule from the registry and run
    the full pipeline in one call.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any

import torch

from qvartools.diag import (
    DiversityConfig,
    DiversitySelector,
)
from qvartools.flows import (
    DiscreteFlowSampler,
    ParticleConservingFlowSampler,
    PhysicsGuidedConfig,
    PhysicsGuidedFlowTrainer,
    verify_particle_conservation,
)
from qvartools.krylov import (
    FlowGuidedKrylovDiag,
    ResidualBasedExpander,
    ResidualExpansionConfig,
    SelectedCIExpander,
    SKQDConfig,
)
from qvartools.molecules import get_molecule
from qvartools.nqs import DenseNQS
from qvartools.pipeline_config import PipelineConfig

__all__ = [
    "PipelineConfig",
    "FlowGuidedKrylovPipeline",
    "run_molecular_benchmark",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class FlowGuidedKrylovPipeline:
    """Main orchestrator for the flow-guided Krylov / SQD pipeline.

    Supports three subspace diagonalization modes via ``config.subspace_mode``:

    - ``"classical_krylov"``: Classical exact time evolution (no Trotter error).
    - ``"skqd"``: Real SKQD via quantum circuit Trotterized evolution (CUDA-Q).
    - ``"sqd"``: IBM SQD sampling-based batch diagonalization.

    When ``config.skip_nf_training`` is ``True``, operates in **Direct-CI
    mode**: generates HF + singles + doubles deterministically, then proceeds
    directly to subspace diagonalization.

    Parameters
    ----------
    hamiltonian : MolecularHamiltonian or Hamiltonian
        The system Hamiltonian.
    config : PipelineConfig
        Pipeline hyperparameters.
    exact_energy : float or None, optional
        Known exact (FCI) energy for error reporting.
    auto_adapt : bool, optional
        If ``True``, automatically scale the config to the system size.

    Examples
    --------
    >>> pipeline = FlowGuidedKrylovPipeline(hamiltonian, PipelineConfig())
    >>> results = pipeline.run(progress=False)
    >>> "final_energy" in results
    True
    """

    def __init__(
        self,
        hamiltonian: Any,
        config: PipelineConfig | None = None,
        exact_energy: float | None = None,
        auto_adapt: bool = True,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.exact_energy = exact_energy
        self.config = config or PipelineConfig()

        # Detect molecular Hamiltonian
        self._is_molecular: bool = hasattr(hamiltonian, "integrals")

        # Compute Hilbert-space size
        if self._is_molecular:
            integrals = hamiltonian.integrals
            n_orbitals = integrals.n_orbitals
            n_alpha = integrals.n_alpha
            n_beta = integrals.n_beta
            self._n_orbitals: int = n_orbitals
            self._n_alpha: int = n_alpha
            self._n_beta: int = n_beta
            self._n_valid_configs: int = math.comb(n_orbitals, n_alpha) * math.comb(
                n_orbitals, n_beta
            )
        else:
            self._n_orbitals = hamiltonian.num_sites
            self._n_alpha = 0
            self._n_beta = 0
            self._n_valid_configs = 2**hamiltonian.num_sites

        # Adapt config to system size
        if auto_adapt:
            self.config = self.config.adapt_to_system_size(self._n_valid_configs)

        # Initialize components
        self.flow: torch.nn.Module
        self.nqs: torch.nn.Module
        self.reference_state: torch.Tensor
        self.trainer: PhysicsGuidedFlowTrainer | None = None
        self._essential_configs: torch.Tensor | None = None
        self.nf_basis: torch.Tensor | None = None
        self.results: dict[str, Any] = {}
        self._init_components()

        logger.info(
            "Pipeline initialized: num_sites=%d, valid_configs=%d, "
            "is_molecular=%s, subspace_mode=%s",
            hamiltonian.num_sites,
            self._n_valid_configs,
            self._is_molecular,
            self.config.subspace_mode,
        )

    def _init_components(self) -> None:
        """Create the flow sampler, NQS, and reference state.

        Initialises ``self.flow``, ``self.nqs``, and
        ``self.reference_state`` based on the pipeline configuration
        and whether the Hamiltonian is molecular.
        """
        num_sites = self.hamiltonian.num_sites
        cfg = self.config
        device = torch.device(cfg.device)

        # --- Flow sampler ---
        if self._is_molecular and cfg.use_particle_conserving_flow:
            self.flow = ParticleConservingFlowSampler(
                num_sites=num_sites,
                n_alpha=self._n_alpha,
                n_beta=self._n_beta,
                hidden_dims=list(cfg.nf_hidden_dims),
            ).to(device)
        else:
            self.flow = DiscreteFlowSampler(
                num_sites=num_sites,
                hidden_dims=list(cfg.nf_hidden_dims),
            ).to(device)

        # --- NQS ---
        self.nqs = self._create_nqs(num_sites, cfg, device)

        # --- Reference state ---
        if self._is_molecular:
            ref = torch.zeros(num_sites, dtype=torch.float32, device=device)
            ref[: self._n_alpha] = 1.0
            ref[self._n_orbitals : self._n_orbitals + self._n_beta] = 1.0
            self.reference_state = ref
        else:
            self.reference_state = torch.zeros(
                num_sites, dtype=torch.float32, device=device
            )

    # ------------------------------------------------------------------
    # NQS factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_nqs(
        num_sites: int,
        cfg: PipelineConfig,
        device: torch.device,
    ) -> torch.nn.Module:
        """Instantiate the NQS model based on ``cfg.nqs_type``.

        Parameters
        ----------
        num_sites : int
            Number of qubit / lattice sites.
        cfg : PipelineConfig
            Pipeline configuration (uses ``nqs_type`` and ``nqs_hidden_dims``).
        device : torch.device
            Target device.

        Returns
        -------
        torch.nn.Module
            The NQS model, moved to *device*.
        """
        hidden = list(cfg.nqs_hidden_dims)
        nqs_type = cfg.nqs_type

        if nqs_type == "dense":
            return DenseNQS(num_sites=num_sites, hidden_dims=hidden).to(device)

        if nqs_type == "signed":
            from qvartools.nqs import SignedDenseNQS

            return SignedDenseNQS(num_sites=num_sites, hidden_dims=hidden).to(device)

        if nqs_type == "complex":
            from qvartools.nqs import ComplexNQS

            return ComplexNQS(num_sites=num_sites, hidden_dims=hidden).to(device)

        if nqs_type == "rbm":
            from qvartools.nqs import RBMQuantumState

            return RBMQuantumState(num_sites=num_sites).to(device)

        if nqs_type == "transformer":
            from qvartools.nqs.adapters import TransformerAsNQS
            from qvartools.nqs.transformer.autoregressive import (
                AutoregressiveTransformer,
            )

            n_orb = num_sites // 2
            transformer = AutoregressiveTransformer(
                n_orbitals=n_orb,
                n_alpha=n_orb // 2,
                n_beta=n_orb // 2,
                embed_dim=hidden[0] if hidden else 64,
            )
            return TransformerAsNQS(transformer).to(device)

        logger.warning("Unknown nqs_type %r, falling back to DenseNQS.", nqs_type)
        return DenseNQS(num_sites=num_sites, hidden_dims=hidden).to(device)

    # ------------------------------------------------------------------
    # Essential config generation (Direct-CI)
    # ------------------------------------------------------------------

    def _generate_essential_configs(self) -> torch.Tensor:
        """Generate HF + singles + doubles for Direct-CI mode.

        Returns
        -------
        torch.Tensor
            Unique essential configurations, shape ``(n_configs, num_sites)``.
        """
        n_orb = self._n_orbitals
        n_alpha = self._n_alpha
        n_beta = self._n_beta
        device = torch.device(self.config.device)

        hf_state = self.reference_state.clone()
        essential = [hf_state.clone()]

        occ_alpha = list(range(n_alpha))
        occ_beta = list(range(n_beta))
        virt_alpha = list(range(n_alpha, n_orb))
        virt_beta = list(range(n_beta, n_orb))

        # Single excitations
        for i in occ_alpha:
            for a in virt_alpha:
                cfg = hf_state.clone()
                cfg[i] = 0
                cfg[a] = 1
                essential.append(cfg)

        for i in occ_beta:
            for a in virt_beta:
                cfg = hf_state.clone()
                cfg[i + n_orb] = 0
                cfg[a + n_orb] = 1
                essential.append(cfg)

        # Double excitations (capped for large systems)
        max_doubles = 5000
        doubles_count = 0

        for i, j in combinations(occ_alpha, 2):
            for a, b in combinations(virt_alpha, 2):
                if doubles_count >= max_doubles:
                    break
                cfg = hf_state.clone()
                cfg[i] = 0
                cfg[j] = 0
                cfg[a] = 1
                cfg[b] = 1
                essential.append(cfg)
                doubles_count += 1
            if doubles_count >= max_doubles:
                break

        for i, j in combinations(occ_beta, 2):
            for a, b in combinations(virt_beta, 2):
                if doubles_count >= max_doubles:
                    break
                cfg = hf_state.clone()
                cfg[i + n_orb] = 0
                cfg[j + n_orb] = 0
                cfg[a + n_orb] = 1
                cfg[b + n_orb] = 1
                essential.append(cfg)
                doubles_count += 1
            if doubles_count >= max_doubles:
                break

        # Alpha-beta doubles (most important for correlation)
        for i in occ_alpha:
            for j in occ_beta:
                for a in virt_alpha:
                    for b in virt_beta:
                        if doubles_count >= max_doubles:
                            break
                        cfg = hf_state.clone()
                        cfg[i] = 0
                        cfg[j + n_orb] = 0
                        cfg[a] = 1
                        cfg[b + n_orb] = 1
                        essential.append(cfg)
                        doubles_count += 1
                    if doubles_count >= max_doubles:
                        break
                if doubles_count >= max_doubles:
                    break
            if doubles_count >= max_doubles:
                break

        essential_tensor = torch.stack(essential).to(device)
        essential_tensor = torch.unique(essential_tensor, dim=0)

        logger.info(
            "Generated %d essential configs (HF + singles + doubles)",
            essential_tensor.shape[0],
        )
        return essential_tensor

    # ------------------------------------------------------------------
    # Stage 1: Train flow + NQS (or generate Direct-CI basis)
    # ------------------------------------------------------------------

    def train_flow_nqs(self, progress: bool = True) -> dict[str, Any]:
        """Stage 1: Physics-guided joint training of the flow and NQS.

        If ``config.skip_nf_training`` is ``True``, generates essential
        configs (HF + singles + doubles) directly without NF training.

        Parameters
        ----------
        progress : bool, optional
            If ``True`` (default), log training progress.

        Returns
        -------
        dict
            Training history with loss/energy lists, or
            ``{"energies": [], "skipped": True}`` in Direct-CI mode.
        """
        cfg = self.config

        # Direct-CI mode: skip NF training
        if cfg.skip_nf_training and self._is_molecular:
            logger.info("Direct-CI mode: generating essential configs.")
            self._essential_configs = self._generate_essential_configs()
            self.results["training_history"] = {"energies": [], "skipped": True}
            self.results["nf_nqs_energy"] = None
            self.results["skip_nf_training"] = True
            return {"energies": [], "skipped": True}

        physics_config = PhysicsGuidedConfig(
            samples_per_batch=cfg.samples_per_batch,
            num_batches=cfg.num_batches,
            num_epochs=cfg.max_epochs,
            min_epochs=cfg.min_epochs,
            convergence_threshold=cfg.convergence_threshold,
            flow_lr=cfg.flow_lr,
            nqs_lr=cfg.nqs_lr,
            teacher_weight=cfg.teacher_weight,
            physics_weight=cfg.physics_weight,
            entropy_weight=cfg.entropy_weight,
            device=cfg.device,
        )

        self.trainer = PhysicsGuidedFlowTrainer(
            flow=self.flow,
            nqs=self.nqs,
            hamiltonian=self.hamiltonian,
            config=physics_config,
            device=cfg.device,
        )

        logger.info("Stage 1: Starting physics-guided flow+NQS training.")
        history = self.trainer.train(progress=progress)

        basis_size = 0
        if self.trainer.accumulated_basis is not None:
            basis_size = self.trainer.accumulated_basis.shape[0]
        logger.info(
            "Stage 1 complete: %d epochs, %d accumulated configs.",
            len(history.get("total_loss", [])),
            basis_size,
        )

        self.results["training_history"] = history
        if history.get("energies"):
            self.results["nf_nqs_energy"] = history["energies"][-1]

        return history

    # ------------------------------------------------------------------
    # Stage 2: Basis extraction and diversity selection
    # ------------------------------------------------------------------

    def extract_and_select_basis(self) -> torch.Tensor:
        """Stage 2: Extract the accumulated basis and apply diversity selection.

        In Direct-CI mode, uses essential configs directly.

        Returns
        -------
        torch.Tensor
            Selected basis configurations.
        """
        cfg = self.config

        # Direct-CI mode: use essential configs directly
        if cfg.skip_nf_training and self._essential_configs is not None:
            selected_basis = self._essential_configs
            self.nf_basis = selected_basis
            self.results["nf_basis_size"] = selected_basis.shape[0]
            self.results["diversity_stats"] = {"skipped": True}
            logger.info(
                "Stage 2: Direct-CI mode, %d essential configs.",
                selected_basis.shape[0],
            )
            return selected_basis

        # Get accumulated basis from training
        if self.trainer is None or self.trainer.accumulated_basis is None:
            raise RuntimeError(
                "No accumulated basis found. Run train_flow_nqs() first."
            )

        basis = self.trainer.accumulated_basis.clone()
        logger.info("Stage 2: Extracted %d accumulated configurations.", basis.shape[0])

        # Verify particle conservation
        if self._is_molecular:
            is_valid, stats = verify_particle_conservation(
                basis,
                n_orbitals=self._n_orbitals,
                n_alpha=self._n_alpha,
                n_beta=self._n_beta,
            )
            if not is_valid:
                alpha_part = basis[:, : self._n_orbitals]
                beta_part = basis[:, self._n_orbitals :]
                alpha_ok = alpha_part.sum(dim=1) == self._n_alpha
                beta_ok = beta_part.sum(dim=1) == self._n_beta
                valid_mask = alpha_ok & beta_ok
                basis = basis[valid_mask]
                logger.info("After filtering: %d valid configurations.", basis.shape[0])

        # Apply diversity selection
        if cfg.use_diversity_selection and basis.shape[0] > 0:
            remaining_frac = 1.0 - (0.05 + 0.15 + cfg.rank_2_fraction + 0.25)
            remaining_frac = max(remaining_frac, 0.0)

            diversity_config = DiversityConfig(
                max_configs=cfg.max_diverse_configs,
                rank_2_fraction=cfg.rank_2_fraction,
                rank_4_plus_fraction=remaining_frac,
            )

            selector = DiversitySelector(
                config=diversity_config,
                reference=self.reference_state.cpu(),
                n_orbitals=self.hamiltonian.num_sites,
            )

            selected, selection_stats = selector.select(basis.cpu())
            logger.info(
                "Stage 2: Diversity selection: %d -> %d configs.",
                basis.shape[0],
                selected.shape[0],
            )
            self.results["diversity_stats"] = selection_stats

            # Always include essential configs if available
            if (
                hasattr(self, "trainer")
                and self.trainer is not None
                and hasattr(self.trainer, "_essential_configs")
                and self.trainer._essential_configs is not None
            ):
                essential = self.trainer._essential_configs
                combined = torch.cat([essential.to(selected.device), selected], dim=0)
                selected = torch.unique(combined, dim=0)

            self.nf_basis = selected.to(cfg.device)
            self.results["nf_basis_size"] = self.nf_basis.shape[0]
            return self.nf_basis

        self.nf_basis = basis
        self.results["nf_basis_size"] = basis.shape[0]
        return basis

    # ------------------------------------------------------------------
    # Stage 3: Subspace diagonalization (routing)
    # ------------------------------------------------------------------

    def run_subspace_diag(self, progress: bool = True) -> dict[str, Any]:
        """Stage 3: Subspace diagonalization via SKQD, SKQD-Quantum, or SQD.

        Routes to the appropriate backend based on ``config.subspace_mode``.

        Parameters
        ----------
        progress : bool, optional
            If ``True`` (default), log diagonalization progress.

        Returns
        -------
        dict
            Backend-specific results dictionary.  Always populates
            ``self.results["combined_energy"]``.

        Raises
        ------
        RuntimeError
            If no basis is available (call
            :meth:`extract_and_select_basis` first).
        """
        cfg = self.config
        nf_basis = self.nf_basis

        if nf_basis is None:
            raise RuntimeError(
                "No basis available. Run extract_and_select_basis() first."
            )

        _VALID_MODES = {"sqd", "skqd", "skqd_quantum", "classical_krylov"}
        if cfg.subspace_mode not in _VALID_MODES:
            logger.warning(
                "Unknown subspace_mode %r, falling back to 'classical_krylov'. "
                "Valid modes: %s",
                cfg.subspace_mode,
                ", ".join(sorted(_VALID_MODES)),
            )

        if cfg.subspace_mode == "sqd":
            return self._run_sqd(nf_basis, progress)
        elif cfg.subspace_mode in ("skqd", "skqd_quantum"):
            return self._run_skqd_quantum(nf_basis, progress)
        else:
            return self._run_classical_krylov(nf_basis, progress)

    def _run_classical_krylov(
        self, basis: torch.Tensor, progress: bool = True
    ) -> dict[str, Any]:
        """Run classical Krylov diagonalization (exact time evolution).

        Parameters
        ----------
        basis : torch.Tensor
            NF basis configurations, shape ``(n_basis, num_sites)``.
        progress : bool, optional
            If ``True`` (default), log per-step progress.

        Returns
        -------
        dict
            SKQD results including energy trajectory and basis configs.
        """
        cfg = self.config

        if cfg.skip_skqd or cfg.max_krylov_dim <= 0:
            return self._direct_diagonalize(basis)

        # Compute spectral-range dt if auto_time_step enabled
        time_step = cfg.time_step
        if cfg.auto_time_step and self._is_molecular:
            try:
                from qvartools.krylov.circuits.spectral import (
                    compute_optimal_dt,
                )

                optimal_dt, spectral_range = compute_optimal_dt(self.hamiltonian)
                time_step = optimal_dt
                logger.info(
                    "Auto time step: dt=%.6f (spectral range: %.4f Ha)",
                    optimal_dt,
                    spectral_range,
                )
            except Exception:
                logger.info("Spectral range unavailable, using dt=%.4f", time_step)

        skqd_config = SKQDConfig(
            max_krylov_dim=cfg.max_krylov_dim,
            time_step=time_step,
            shots_per_krylov=cfg.shots_per_krylov,
            regularization=cfg.skqd_regularization,
        )

        skqd = FlowGuidedKrylovDiag(
            hamiltonian=self.hamiltonian,
            config=skqd_config,
            nf_basis=basis.cpu().long(),
        )

        results = skqd.run_with_nf(progress=progress)

        skqd_energy = results.get("energy", float("inf"))

        self.results["skqd_results"] = results
        self.results["skqd_energy"] = skqd_energy
        self.results["combined_energy"] = skqd_energy
        return results

    def _run_skqd_quantum(
        self, basis: torch.Tensor, progress: bool = True
    ) -> dict[str, Any]:
        """Run quantum circuit SKQD (Trotterized evolution via CUDA-Q).

        Parameters
        ----------
        basis : torch.Tensor
            NF basis configurations (unused in quantum mode; kept for
            API consistency).
        progress : bool, optional
            If ``True`` (default), log per-step progress.

        Returns
        -------
        dict
            Quantum SKQD results.

        Raises
        ------
        ImportError
            If ``cudaq`` or ``cupy`` packages are not installed.
        """
        cfg = self.config

        try:
            from qvartools.krylov.circuits.circuit_skqd import (
                QuantumCircuitSKQD,
                QuantumSKQDConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "Quantum SKQD requires cudaq and cupy. "
                "Install with: pip install cudaq cupy-cuda12x"
            ) from exc

        # Compute spectral-range dt if auto_time_step enabled
        evolution_time = cfg.quantum_total_evolution_time
        if cfg.auto_time_step and self._is_molecular:
            try:
                from qvartools.krylov.circuits.spectral import (
                    compute_optimal_dt,
                )

                optimal_dt, _sr = compute_optimal_dt(self.hamiltonian)
                evolution_time = optimal_dt
            except Exception:
                pass

        quantum_config = QuantumSKQDConfig(
            max_krylov_dim=cfg.max_krylov_dim,
            total_evolution_time=evolution_time,
            num_trotter_steps=cfg.quantum_num_trotter_steps,
            shots=cfg.quantum_shots,
            cudaq_target=cfg.quantum_cudaq_target,
            cudaq_option=cfg.quantum_cudaq_option,
            initial_state="hf",
        )

        solver = QuantumCircuitSKQD.from_molecular_hamiltonian(
            self.hamiltonian, config=quantum_config
        )

        results = solver.run(progress=progress)

        quantum_energy = results["best_energy"]
        self.results["quantum_skqd_results"] = results
        self.results["quantum_skqd_energy"] = quantum_energy
        self.results["combined_energy"] = quantum_energy
        return results

    def _run_sqd(self, basis: torch.Tensor, progress: bool = True) -> dict[str, Any]:
        """Run SQD (sampling-based batch diagonalization).

        Parameters
        ----------
        basis : torch.Tensor
            Configuration basis, shape ``(n_basis, num_sites)``.
        progress : bool, optional
            If ``True`` (default), log progress.

        Returns
        -------
        dict
            SQD results including best energy and batch statistics.
        """
        cfg = self.config

        try:
            from qvartools.krylov.circuits.sqd import (
                SQDConfig as SQDPipelineConfig,
            )
            from qvartools.krylov.circuits.sqd import (
                SQDSolver as SQDPipelineSolver,
            )
        except ImportError:
            # Fallback to direct diagonalization
            logger.warning("SQD pipeline not available, falling back to direct diag.")
            return self._direct_diagonalize(basis)

        enable_recovery = cfg.sqd_noise_rate > 0

        sqd_config = SQDPipelineConfig(
            num_batches=cfg.sqd_num_batches,
            batch_size=cfg.sqd_batch_size,
            self_consistent_iters=cfg.sqd_self_consistent_iters,
            spin_penalty=cfg.sqd_spin_penalty,
            noise_rate=cfg.sqd_noise_rate,
            enable_config_recovery=enable_recovery,
            use_spin_symmetry_enhancement=cfg.sqd_use_spin_symmetry,
        )

        solver = SQDPipelineSolver(
            hamiltonian=self.hamiltonian,
            config=sqd_config,
        )

        results = solver.run(basis)

        sqd_energy = results["energy"]
        self.results["sqd_results"] = results
        self.results["sqd_energy"] = sqd_energy
        self.results["combined_energy"] = sqd_energy
        return results

    def _direct_diagonalize(self, basis: torch.Tensor) -> dict[str, Any]:
        """Compute energy by direct diagonalization of the basis.

        Parameters
        ----------
        basis : torch.Tensor
            Configuration basis, shape ``(n_basis, num_sites)``.

        Returns
        -------
        dict
            Results with key ``"energies_combined"`` and
            ``"direct_diag": True``.
        """
        if basis.shape[0] == 0:
            self.results["combined_energy"] = float("nan")
            return {"energies_combined": [float("nan")], "direct_diag": True}

        from qvartools.krylov.expansion.residual_expansion import (
            _diagonalise_in_basis,
        )

        energy, _ = _diagonalise_in_basis(self.hamiltonian, basis)
        self.results["combined_energy"] = energy
        return {"energies_combined": [energy], "direct_diag": True}

    # ------------------------------------------------------------------
    # Stage 3 (legacy): Residual / perturbative basis expansion
    # ------------------------------------------------------------------

    def run_residual_expansion(self, basis: torch.Tensor) -> torch.Tensor:
        """Expand the basis via residual or perturbative selection.

        This is the legacy Stage 3 from the 4-stage pipeline.  When using
        ``run()`` or ``run_subspace_diag()``, this is NOT called -- those
        methods route directly to the appropriate subspace diag backend.

        Parameters
        ----------
        basis : torch.Tensor
            Current basis configurations, shape ``(n_basis, num_sites)``.

        Returns
        -------
        torch.Tensor
            Expanded basis configurations on ``config.device``.
        """
        if not self.config.use_residual_expansion:
            return basis

        if basis.shape[0] == 0:
            return basis

        from qvartools.krylov.expansion.residual_expansion import (
            _diagonalise_in_basis,
        )

        energy, eigenvector = _diagonalise_in_basis(self.hamiltonian, basis)

        expansion_config = ResidualExpansionConfig(
            max_configs_per_iter=self.config.residual_configs_per_iter,
            residual_threshold=self.config.residual_threshold,
            max_iterations=self.config.residual_iterations,
            max_basis_size=self.config.max_accumulated_basis,
        )

        if self.config.use_perturbative_selection:
            expander: Any = SelectedCIExpander(
                hamiltonian=self.hamiltonian,
                config=expansion_config,
            )
        else:
            expander = ResidualBasedExpander(
                hamiltonian=self.hamiltonian,
                config=expansion_config,
            )

        expanded_basis, expansion_stats = expander.expand_basis(
            current_basis=basis,
            energy=energy,
            eigenvector=eigenvector,
        )

        return expanded_basis.to(self.config.device)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, progress: bool = True) -> dict[str, Any]:
        """Execute the complete pipeline.

        Runs training (or Direct-CI), basis extraction, and subspace
        diagonalization in sequence.

        Parameters
        ----------
        progress : bool, optional
            If ``True`` (default), log progress for all stages.

        Returns
        -------
        dict
            Aggregated results dictionary with keys including
            ``"final_energy"``, ``"nf_basis_size"``, ``"error_mha"``
            (when ``exact_energy`` is set), and stage-specific sub-dicts.
        """
        logger.info("=" * 60)
        logger.info("Starting FlowGuidedKrylovPipeline")
        logger.info("=" * 60)

        # Stage 1: Train flow + NQS (or generate essential configs)
        self.train_flow_nqs(progress=progress)

        # Stage 2: Extract and select basis
        self.extract_and_select_basis()

        # Stage 3: Subspace diagonalization
        self.run_subspace_diag(progress=progress)

        # Summary
        self._print_summary()

        return self.results

    def _print_summary(self) -> None:
        """Log a human-readable results summary via the module logger."""
        mode_label = self.config.subspace_mode.upper()
        if self.config.subspace_mode == "sqd":
            if self.config.sqd_noise_rate > 0:
                mode_label += f" (Recovery, noise={self.config.sqd_noise_rate:.2f})"
            else:
                mode_label += " (Clean)"

        logger.info("=" * 60)
        logger.info("Subspace mode: %s", mode_label)

        if self.results.get("skip_nf_training"):
            logger.info("Mode: Direct-CI (NF training skipped)")

        if "nf_basis_size" in self.results:
            logger.info("Basis size: %d", self.results["nf_basis_size"])

        best_energy = self.results.get("combined_energy")
        if best_energy is not None:
            logger.info("Final energy: %.10f Ha", best_energy)

        if self.exact_energy is not None and best_energy is not None:
            error_mha = (best_energy - self.exact_energy) * 1000.0
            self.results["error_mha"] = error_mha
            logger.info("Error: %.4f mHa", error_mha)
            if abs(error_mha) < 1.6:
                logger.info("Chemical accuracy: PASS")
            else:
                logger.info("Chemical accuracy: FAIL")

        self.results["final_energy"] = best_energy
        self.results["exact_energy"] = self.exact_energy
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Convenience benchmark function
# ---------------------------------------------------------------------------


def run_molecular_benchmark(
    molecule: str,
    config: PipelineConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load a molecule from the registry and run the full pipeline.

    Parameters
    ----------
    molecule : str
        Molecule name (case-insensitive).  Must be a key in
        :data:`~qvartools.molecules.MOLECULE_REGISTRY`.
    config : PipelineConfig or None, optional
        Pipeline hyperparameters.  If ``None``, uses defaults.
    verbose : bool, optional
        If ``True`` (default), print a summary to stdout.

    Returns
    -------
    dict
        Pipeline results dictionary (see
        :meth:`FlowGuidedKrylovPipeline.run`).

    Examples
    --------
    >>> results = run_molecular_benchmark("H2")
    >>> "final_energy" in results
    True
    """
    if config is None:
        config = PipelineConfig()

    hamiltonian, mol_info = get_molecule(molecule, device=config.device)

    if verbose:
        print(f"Molecule: {mol_info['name']}")
        print(f"Qubits:   {mol_info['n_qubits']}")
        print(f"Basis:    {mol_info['basis']}")
        print("-" * 40)

    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=config,
        exact_energy=None,
        auto_adapt=True,
    )

    results = pipeline.run(progress=verbose)

    if verbose:
        print("=" * 40)
        print(f"Final energy: {results.get('final_energy', 'N/A')}")
        if "nf_basis_size" in results:
            print(f"Basis size:   {results['nf_basis_size']}")
        print("=" * 40)

    return results
