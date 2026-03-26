"""
skqd --- Sample-based Krylov Quantum Diagonalization solver
============================================================

Implements :class:`SKQDSolver`, which combines normalizing-flow sampling
with Krylov subspace expansion and residual-based basis enrichment for
ground-state energy estimation.

Pipeline
--------
1. Train a normalizing flow to generate an initial configuration basis.
2. Sample configurations from the trained flow.
3. Run flow-guided SKQD with Krylov subspace methods.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.solvers.solver import Solver, SolverResult

__all__ = [
    "SKQDSolver",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

_DEFAULT_SKQD_CONFIG: dict[str, Any] = {
    "max_krylov_dim": 10,
    "time_step": 0.1,
    "shots_per_krylov": 1000,
    "num_eigenvalues": 1,
    "regularization": 1e-8,
}

_DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "samples_per_batch": 500,
    "num_batches": 10,
    "num_epochs": 200,
    "min_epochs": 50,
    "convergence_threshold": 0.01,
    "flow_lr": 1e-3,
    "nqs_lr": 1e-3,
    "teacher_weight": 1.0,
    "physics_weight": 0.0,
    "entropy_weight": 0.0,
    "initial_temperature": 2.0,
    "final_temperature": 0.1,
    "temperature_decay_epochs": 100,
    "inject_essential_configs": True,
}


# ---------------------------------------------------------------------------
# SKQDSolver
# ---------------------------------------------------------------------------


class SKQDSolver(Solver):
    """Sample-based Krylov quantum diagonalization solver.

    Trains a normalizing flow, samples an initial basis, and runs
    Krylov-based diagonalisation to obtain the ground-state energy.

    Parameters
    ----------
    n_samples : int, optional
        Number of configuration samples from the trained flow
        (default ``5000``).
    skqd_config : dict or None, optional
        Override SKQD hyperparameters forwarded to
        :class:`~qvartools.krylov.SKQDConfig`.
    training_config : dict or None, optional
        Override training hyperparameters forwarded to
        :class:`~qvartools.flows.PhysicsGuidedConfig`.
    device : str, optional
        Torch device (default ``"cpu"``).

    Attributes
    ----------
    n_samples : int
        Post-training sample count.
    skqd_config : dict
        Merged SKQD hyperparameters.
    training_config : dict
        Merged training hyperparameters.
    device : str
        Torch device string.

    Examples
    --------
    >>> solver = SKQDSolver(n_samples=3000)
    >>> result = solver.solve(hamiltonian, mol_info)
    >>> result.method
    'SKQD'
    """

    def __init__(
        self,
        n_samples: int = 5000,
        skqd_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        device: str = "cpu",
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        self.n_samples: int = n_samples
        self.device: str = device

        merged_skqd = dict(_DEFAULT_SKQD_CONFIG)
        if skqd_config is not None:
            merged_skqd.update(skqd_config)
        self.skqd_config: dict[str, Any] = merged_skqd

        merged_train = dict(_DEFAULT_TRAINING_CONFIG)
        if training_config is not None:
            merged_train.update(training_config)
        self.training_config: dict[str, Any] = merged_train

    def solve(self, hamiltonian: Hamiltonian, mol_info: dict[str, Any]) -> SolverResult:
        """Run the SKQD pipeline.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            The molecular Hamiltonian.
        mol_info : dict
            Molecular metadata.  Must contain ``"n_qubits"``.

        Returns
        -------
        SolverResult
            SKQD energy result with Krylov history in metadata.
        """
        from qvartools.flows import (
            PhysicsGuidedConfig,
            PhysicsGuidedFlowTrainer,
        )
        from qvartools.krylov import FlowGuidedKrylovDiag, SKQDConfig
        from qvartools.nqs import DenseNQS

        t_start = time.perf_counter()
        n_qubits = mol_info["n_qubits"]

        # --- Step 1: Create and train flow ---
        flow = self._create_flow(hamiltonian, n_qubits)
        flow = flow.to(self.device)

        nqs = DenseNQS(num_sites=n_qubits, hidden_dims=[128, 64])
        nqs = nqs.to(self.device)

        train_cfg = PhysicsGuidedConfig(
            **{**self.training_config, "device": self.device}
        )
        trainer = PhysicsGuidedFlowTrainer(
            flow=flow, nqs=nqs, hamiltonian=hamiltonian, config=train_cfg
        )
        history = trainer.train(progress=True)

        # --- Step 2: Sample basis ---
        flow.eval()
        with torch.no_grad():
            _, unique_configs = flow.sample(self.n_samples)

        # Merge with accumulated basis
        if trainer.accumulated_basis is not None:
            combined = torch.cat(
                [unique_configs.to(self.device), trainer.accumulated_basis],
                dim=0,
            )
            nf_basis = torch.unique(combined, dim=0).cpu()
        else:
            nf_basis = unique_configs.cpu()

        # --- Step 3: Run flow-guided SKQD ---
        skqd_cfg = SKQDConfig(**self.skqd_config)
        skqd = FlowGuidedKrylovDiag(
            hamiltonian=hamiltonian,
            config=skqd_cfg,
            nf_basis=nf_basis,
        )
        skqd_result = skqd.run_with_nf(progress=True)

        energy = skqd_result["energy"]
        diag_dim = skqd_result["basis_size"]
        wall_time = time.perf_counter() - t_start

        metadata: dict[str, Any] = {
            "training_history": history,
            "skqd_result": skqd_result,
            "nf_energy": skqd_result.get("nf_energy"),
            "energies_per_step": skqd_result.get("energies_per_step", []),
            "krylov_dim": skqd_result.get("krylov_dim"),
            "n_samples": self.n_samples,
        }

        logger.info(
            "SKQDSolver [%s]: energy=%.10f, basis=%d, time=%.2fs",
            mol_info.get("name", "unknown"),
            energy,
            diag_dim,
            wall_time,
        )

        return SolverResult(
            energy=energy,
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="SKQD",
            converged=True,
            metadata=metadata,
        )

    def _create_flow(self, hamiltonian: Hamiltonian, n_qubits: int) -> torch.nn.Module:
        """Instantiate the normalizing flow.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            Used to extract particle numbers for conserving flows.
        n_qubits : int
            Number of qubits / spin-orbitals.

        Returns
        -------
        torch.nn.Module
            The flow sampler instance.
        """
        if hasattr(hamiltonian, "integrals"):
            from qvartools.flows import ParticleConservingFlowSampler

            integrals = hamiltonian.integrals
            return ParticleConservingFlowSampler(
                num_sites=n_qubits,
                n_alpha=integrals.n_alpha,
                n_beta=integrals.n_beta,
            )

        from qvartools.flows import DiscreteFlowSampler

        return DiscreteFlowSampler(num_sites=n_qubits)
