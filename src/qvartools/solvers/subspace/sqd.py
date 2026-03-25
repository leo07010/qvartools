"""
sqd --- Sample-based Quantum Diagonalization solver
====================================================

Implements :class:`SQDSolver`, which trains a normalizing flow to sample
computational-basis configurations, builds a projected Hamiltonian in the
sampled basis, and diagonalises it to obtain the ground-state energy.

Pipeline
--------
1. Initialise flow (particle-conserving or discrete) and NQS.
2. Train the flow via physics-guided training.
3. Sample configurations from the trained flow.
4. Build the projected Hamiltonian in the sampled basis.
5. Diagonalise to obtain ground-state energy.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.solvers.solver import Solver, SolverResult

__all__ = [
    "SQDSolver",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------------

_DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
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
# SQDSolver
# ---------------------------------------------------------------------------


class SQDSolver(Solver):
    """Sample-based quantum diagonalization solver.

    Trains a normalizing flow to generate computational-basis
    configurations, builds a projected Hamiltonian in the sampled basis,
    and diagonalises it.

    Parameters
    ----------
    n_samples : int, optional
        Number of configuration samples to draw after training
        (default ``5000``).
    flow_type : str, optional
        Type of normalizing flow: ``"particle_conserving"`` or
        ``"discrete"`` (default ``"particle_conserving"``).
    training_config : dict or None, optional
        Override training hyperparameters.  Keys are forwarded to
        :class:`~qvartools.flows.PhysicsGuidedConfig`.  If ``None``,
        sensible defaults are used.
    device : str, optional
        Torch device for computation (default ``"cpu"``).

    Attributes
    ----------
    n_samples : int
        Number of post-training samples.
    flow_type : str
        Flow architecture identifier.
    training_config : dict
        Merged training hyperparameters.
    device : str
        Torch device string.

    Examples
    --------
    >>> solver = SQDSolver(n_samples=3000)
    >>> result = solver.solve(hamiltonian, mol_info)
    >>> result.method
    'SQD'
    """

    def __init__(
        self,
        n_samples: int = 5000,
        flow_type: str = "particle_conserving",
        training_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if flow_type not in ("particle_conserving", "discrete"):
            raise ValueError(
                f"flow_type must be 'particle_conserving' or 'discrete', "
                f"got {flow_type!r}"
            )

        self.n_samples: int = n_samples
        self.flow_type: str = flow_type
        self.device: str = device

        merged = dict(_DEFAULT_TRAINING_CONFIG)
        if training_config is not None:
            merged.update(training_config)
        self.training_config: Dict[str, Any] = merged

    def solve(
        self, hamiltonian: Hamiltonian, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Run the SQD pipeline.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            The molecular Hamiltonian.
        mol_info : dict
            Molecular metadata.  Must contain ``"n_qubits"``.

        Returns
        -------
        SolverResult
            SQD energy result with training history in metadata.
        """
        from qvartools.flows import (
            PhysicsGuidedConfig,
            PhysicsGuidedFlowTrainer,
        )
        from qvartools.nqs import DenseNQS
        from qvartools.diag import (
            ProjectedHamiltonianBuilder,
            compute_ground_state_energy,
        )

        t_start = time.perf_counter()
        n_qubits = mol_info["n_qubits"]

        # --- Step 1: Create flow ---
        flow = self._create_flow(hamiltonian, n_qubits)
        flow = flow.to(self.device)

        # --- Step 2: Create NQS ---
        nqs = DenseNQS(num_sites=n_qubits, hidden_dims=[128, 64])
        nqs = nqs.to(self.device)

        # --- Step 3: Train ---
        train_cfg = PhysicsGuidedConfig(
            **{**self.training_config, "device": self.device}
        )
        trainer = PhysicsGuidedFlowTrainer(
            flow=flow, nqs=nqs, hamiltonian=hamiltonian, config=train_cfg
        )
        history = trainer.train(progress=True)

        # --- Step 4: Sample configurations ---
        flow.eval()
        with torch.no_grad():
            all_configs, unique_configs = flow.sample(self.n_samples)

        # Merge with accumulated basis from training
        if trainer.accumulated_basis is not None:
            combined = torch.cat(
                [unique_configs.to(self.device), trainer.accumulated_basis],
                dim=0,
            )
            basis = torch.unique(combined, dim=0)
        else:
            basis = unique_configs.to(self.device)

        # --- Step 5: Build projected H and diagonalise ---
        builder = ProjectedHamiltonianBuilder(hamiltonian)
        h_proj = builder.build(basis.cpu())

        energy = compute_ground_state_energy(h_proj)
        diag_dim = basis.shape[0]

        wall_time = time.perf_counter() - t_start

        metadata: Dict[str, Any] = {
            "training_history": history,
            "basis_size": diag_dim,
            "n_samples": self.n_samples,
            "flow_type": self.flow_type,
        }

        logger.info(
            "SQDSolver [%s]: energy=%.10f, basis=%d, time=%.2fs",
            mol_info.get("name", "unknown"),
            energy,
            diag_dim,
            wall_time,
        )

        return SolverResult(
            energy=energy,
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="SQD",
            converged=True,
            metadata=metadata,
        )

    def _create_flow(
        self, hamiltonian: Hamiltonian, n_qubits: int
    ) -> torch.nn.Module:
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
        if (
            self.flow_type == "particle_conserving"
            and hasattr(hamiltonian, "integrals")
        ):
            from qvartools.flows import ParticleConservingFlowSampler

            integrals = hamiltonian.integrals
            return ParticleConservingFlowSampler(
                num_sites=n_qubits,
                n_alpha=integrals.n_alpha,
                n_beta=integrals.n_beta,
            )

        from qvartools.flows import DiscreteFlowSampler

        return DiscreteFlowSampler(num_sites=n_qubits)
