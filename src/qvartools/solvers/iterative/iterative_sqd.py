"""
iterative_sqd --- Iterative NF-SQD solver with eigenvector feedback
====================================================================

Implements :class:`IterativeNFSQDSolver`, which couples normalizing-flow
sampling with projected quantum diagonalization (SQD) in a self-consistent
feedback loop.

Classes
-------
IterativeNFSQDSolver
    Iterative normalizing-flow SQD with eigenvector feedback.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

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
    "IterativeNFSQDSolver",
]

logger = logging.getLogger(__name__)


class IterativeNFSQDSolver(Solver):
    """Iterative normalizing-flow SQD with eigenvector feedback.

    At each iteration the solver:

    1. Trains a normalizing flow + NQS to generate configuration samples.
    2. Builds and diagonalises the projected Hamiltonian in the sampled
       basis (SQD).
    3. Uses the ground-state eigenvector coefficients to bias the NQS
       log-amplitudes for the next iteration.

    The feedback loop progressively focuses sampling on the most
    important configurations for the ground state.

    Parameters
    ----------
    n_iterations : int, optional
        Number of outer SQD iterations (default ``5``).
    n_samples : int, optional
        Samples per iteration from the trained flow (default ``2000``).
    training_config : dict or None, optional
        Override training hyperparameters forwarded to
        :class:`~qvartools.flows.PhysicsGuidedConfig`.
    convergence_tol : float, optional
        Stop iterating when the energy change between consecutive
        iterations is below this threshold (default ``1e-6``).
    device : str, optional
        Torch device (default ``"cpu"``).

    Attributes
    ----------
    n_iterations : int
    n_samples : int
    training_config : dict
    convergence_tol : float
    device : str

    Examples
    --------
    >>> solver = IterativeNFSQDSolver(n_iterations=3)
    >>> result = solver.solve(hamiltonian, mol_info)
    >>> result.method
    'IterativeNFSQD'
    """

    def __init__(
        self,
        n_iterations: int = 5,
        n_samples: int = 2000,
        training_config: Optional[Dict[str, Any]] = None,
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

        merged = dict(_DEFAULT_TRAINING_CONFIG)
        if training_config is not None:
            merged.update(training_config)
        self.training_config: Dict[str, Any] = merged

    def solve(
        self, hamiltonian: Hamiltonian, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Run the iterative NF-SQD pipeline.

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
        from qvartools.flows import (
            PhysicsGuidedConfig,
            PhysicsGuidedFlowTrainer,
        )
        from qvartools.nqs import DenseNQS
        from qvartools.diag import (
            ProjectedHamiltonianBuilder,
            solve_generalized_eigenvalue,
        )

        t_start = time.perf_counter()
        n_qubits = mol_info["n_qubits"]

        energies: List[float] = []
        basis_sizes: List[int] = []
        best_energy = float("inf")
        accumulated_basis: Optional[torch.Tensor] = None
        prev_eigvec: Optional[np.ndarray] = None

        converged = False

        for iteration in range(self.n_iterations):
            logger.info(
                "IterativeNFSQD iteration %d / %d",
                iteration + 1,
                self.n_iterations,
            )

            # --- Create fresh flow and NQS each iteration ---
            flow = _create_flow(hamiltonian, n_qubits).to(self.device)
            nqs = DenseNQS(num_sites=n_qubits, hidden_dims=[128, 64])
            nqs = nqs.to(self.device)

            # --- Bias NQS with previous eigenvector if available ---
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

            # Merge with trainer basis and accumulated basis
            parts = [unique_configs]
            if trainer.accumulated_basis is not None:
                parts.append(trainer.accumulated_basis)
            if accumulated_basis is not None:
                parts.append(accumulated_basis)

            combined = torch.cat(parts, dim=0).detach()
            accumulated_basis = torch.unique(combined, dim=0)

            # --- Build projected H and diagonalise ---
            builder = ProjectedHamiltonianBuilder(hamiltonian)
            h_proj = builder.build(accumulated_basis.cpu())

            s_proj = scipy.sparse.eye(
                accumulated_basis.shape[0], format="csr"
            )
            eig_vals, eig_vecs = solve_generalized_eigenvalue(
                h_proj, s_proj, k=1
            )

            energy = float(eig_vals[0])
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

        metadata: Dict[str, Any] = {
            "energies_per_iteration": energies,
            "basis_sizes_per_iteration": basis_sizes,
            "n_iterations_run": len(energies),
            "n_samples": self.n_samples,
        }

        logger.info(
            "IterativeNFSQDSolver [%s]: energy=%.10f, iterations=%d, time=%.2fs",
            mol_info.get("name", "unknown"),
            best_energy,
            len(energies),
            wall_time,
        )

        return SolverResult(
            energy=best_energy,
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="IterativeNFSQD",
            converged=converged,
            metadata=metadata,
        )
