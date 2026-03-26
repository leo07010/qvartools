"""
iterative_skqd --- Iterative NF-SKQD solver with eigenvector feedback
======================================================================

Implements :class:`IterativeNFSKQDSolver`, which couples normalizing-flow
sampling with Krylov subspace quantum diagonalization (SKQD) in a
self-consistent feedback loop.

Classes
-------
IterativeNFSKQDSolver
    Iterative normalizing-flow SKQD with eigenvector feedback.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse
import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.solvers.iterative._utils import (
    _DEFAULT_TRAINING_CONFIG,
    _bias_nqs,
    _create_flow,
)
from qvartools.solvers.solver import Solver, SolverResult

__all__ = [
    "IterativeNFSKQDSolver",
]

logger = logging.getLogger(__name__)


class IterativeNFSKQDSolver(Solver):
    """Iterative normalizing-flow SKQD with eigenvector feedback.

    Same iterative feedback loop as :class:`IterativeNFSQDSolver`, but uses
    Krylov subspace expansion (SKQD) instead of direct projected
    diagonalisation at each iteration.

    Parameters
    ----------
    n_iterations : int, optional
        Number of outer iterations (default ``5``).
    n_samples : int, optional
        Samples per iteration (default ``2000``).
    training_config : dict or None, optional
        Override training hyperparameters.
    skqd_config : dict or None, optional
        Override SKQD hyperparameters forwarded to
        :class:`~qvartools.krylov.SKQDConfig`.
    convergence_tol : float, optional
        Energy convergence threshold (default ``1e-6``).
    device : str, optional
        Torch device (default ``"cpu"``).

    Attributes
    ----------
    n_iterations : int
    n_samples : int
    training_config : dict
    skqd_config : dict
    convergence_tol : float
    device : str

    Examples
    --------
    >>> solver = IterativeNFSKQDSolver(n_iterations=3)
    >>> result = solver.solve(hamiltonian, mol_info)
    >>> result.method
    'IterativeNFSKQD'
    """

    _DEFAULT_SKQD_CONFIG: dict[str, Any] = {
        "max_krylov_dim": 10,
        "time_step": 0.1,
        "shots_per_krylov": 1000,
        "num_eigenvalues": 1,
        "regularization": 1e-8,
    }

    def __init__(
        self,
        n_iterations: int = 5,
        n_samples: int = 2000,
        training_config: dict[str, Any] | None = None,
        skqd_config: dict[str, Any] | None = None,
        convergence_tol: float = 1e-6,
        device: str = "cpu",
    ) -> None:
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        self.n_iterations: int = n_iterations
        self.n_samples: int = n_samples
        self.convergence_tol: float = convergence_tol
        self.device: str = device

        merged_train = dict(_DEFAULT_TRAINING_CONFIG)
        if training_config is not None:
            merged_train.update(training_config)
        self.training_config: dict[str, Any] = merged_train

        merged_skqd = dict(self._DEFAULT_SKQD_CONFIG)
        if skqd_config is not None:
            merged_skqd.update(skqd_config)
        self.skqd_config: dict[str, Any] = merged_skqd

    def solve(self, hamiltonian: Hamiltonian, mol_info: dict[str, Any]) -> SolverResult:
        """Run the iterative NF-SKQD pipeline.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            The molecular Hamiltonian.
        mol_info : dict
            Molecular metadata.  Must contain ``"n_qubits"``.

        Returns
        -------
        SolverResult
            Best energy across iterations with full history in metadata.
        """
        from qvartools.diag import (
            ProjectedHamiltonianBuilder,
            solve_generalized_eigenvalue,
        )
        from qvartools.flows import (
            PhysicsGuidedConfig,
            PhysicsGuidedFlowTrainer,
        )
        from qvartools.krylov import FlowGuidedKrylovDiag, SKQDConfig
        from qvartools.nqs import DenseNQS

        t_start = time.perf_counter()
        n_qubits = mol_info["n_qubits"]

        energies: list[float] = []
        basis_sizes: list[int] = []
        best_energy = float("inf")
        accumulated_basis: torch.Tensor | None = None
        prev_eigvec: np.ndarray | None = None

        converged = False

        for iteration in range(self.n_iterations):
            logger.info(
                "IterativeNFSKQD iteration %d / %d",
                iteration + 1,
                self.n_iterations,
            )

            # --- Create fresh flow and NQS ---
            flow = _create_flow(hamiltonian, n_qubits).to(self.device)
            nqs = DenseNQS(num_sites=n_qubits, hidden_dims=[128, 64])
            nqs = nqs.to(self.device)

            # --- Bias NQS with previous eigenvector ---
            if prev_eigvec is not None and accumulated_basis is not None:
                _bias_nqs(nqs, accumulated_basis, prev_eigvec)

            # --- Train ---
            train_cfg = PhysicsGuidedConfig(
                **{**self.training_config, "device": self.device}
            )
            trainer = PhysicsGuidedFlowTrainer(
                flow=flow, nqs=nqs, hamiltonian=hamiltonian, config=train_cfg
            )
            trainer.train(progress=False)

            # --- Sample ---
            flow.eval()
            with torch.no_grad():
                _, unique_configs = flow.sample(self.n_samples)
            unique_configs = unique_configs.to(self.device)

            parts = [unique_configs]
            if trainer.accumulated_basis is not None:
                parts.append(trainer.accumulated_basis)
            if accumulated_basis is not None:
                parts.append(accumulated_basis)

            nf_basis = torch.unique(torch.cat(parts, dim=0).detach(), dim=0).cpu()

            # --- Run SKQD ---
            skqd_cfg = SKQDConfig(**self.skqd_config)
            skqd = FlowGuidedKrylovDiag(
                hamiltonian=hamiltonian,
                config=skqd_cfg,
                nf_basis=nf_basis,
            )
            skqd_result = skqd.run_with_nf(progress=False)

            energy = skqd_result["energy"]
            accumulated_basis = skqd_result["basis_configs"].to(self.device)

            # Extract eigenvector for feedback (from final projected solve)
            builder = ProjectedHamiltonianBuilder(hamiltonian)
            h_proj = builder.build(accumulated_basis.cpu())
            s_proj = scipy.sparse.eye(accumulated_basis.shape[0], format="csr")
            _, eig_vecs = solve_generalized_eigenvalue(h_proj, s_proj, k=1)
            prev_eigvec = eig_vecs[:, 0]

            energies.append(energy)
            basis_sizes.append(accumulated_basis.shape[0])

            if energy < best_energy:
                best_energy = energy

            logger.info(
                "  Iteration %d: energy=%.10f, basis=%d",
                iteration + 1,
                energy,
                accumulated_basis.shape[0],
            )

            # --- Convergence check ---
            if len(energies) >= 2:
                delta = abs(energies[-1] - energies[-2])
                if delta < self.convergence_tol:
                    logger.info(
                        "Converged at iteration %d (delta=%.2e < %.2e).",
                        iteration + 1,
                        delta,
                        self.convergence_tol,
                    )
                    converged = True
                    break

        wall_time = time.perf_counter() - t_start
        diag_dim = basis_sizes[-1] if basis_sizes else 0

        metadata: dict[str, Any] = {
            "energies_per_iteration": energies,
            "basis_sizes_per_iteration": basis_sizes,
            "n_iterations_run": len(energies),
            "n_samples": self.n_samples,
        }

        logger.info(
            "IterativeNFSKQDSolver [%s]: energy=%.10f, iterations=%d, time=%.2fs",
            mol_info.get("name", "unknown"),
            best_energy,
            len(energies),
            wall_time,
        )

        return SolverResult(
            energy=best_energy,
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="IterativeNFSKQD",
            converged=converged,
            metadata=metadata,
        )
