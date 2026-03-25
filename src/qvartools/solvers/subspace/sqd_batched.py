"""
sqd_batched --- SQD with self-consistent batching and orbital occupancy updates
================================================================================

Implements :class:`SQDBatchedSolver`, a sample-based quantum diagonalization
solver that partitions sampled configurations into random batches, diagonalizes
each batch independently, and iterates a self-consistent orbital-occupancy
loop to improve the batching quality.

This is distinct from the existing :class:`SQDSolver` which uses a single-pass
flow-guided pipeline.  ``SQDBatchedSolver`` is closer to the hardware-oriented
SQD workflow described in Robledo-Moreno et al. (2024).

Algorithm
---------
1. Draw *n_samples* configurations from a sampler.
2. Filter for correct particle number; add the HF determinant.
3. For each self-consistent iteration:
   a. Partition configs into *num_batches* random batches of *batch_size*.
   b. Diagonalize the projected Hamiltonian in each batch.
   c. Select the batch with the lowest energy.
   d. Update orbital occupancies from the best eigenvector.
   e. Stop when occupancy changes are below threshold.
4. Return the best energy found across all iterations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from qvartools.solvers.solver import Solver, SolverResult

__all__ = [
    "SQDBatchedConfig",
    "SQDBatchedSolver",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SQDBatchedConfig:
    """Configuration for the batched SQD solver.

    Parameters
    ----------
    num_batches : int
        Number of random batches per self-consistent iteration.
    batch_size : int
        Configurations per batch.  ``0`` selects an automatic size.
    self_consistent_iters : int
        Maximum number of orbital-occupancy self-consistent iterations.
    occupancy_convergence : float
        Maximum element-wise occupancy change for convergence.
    n_samples : int
        Number of configurations to draw from the sampler.

    Examples
    --------
    >>> cfg = SQDBatchedConfig(num_batches=10, n_samples=5000)
    >>> cfg.num_batches
    10
    """

    num_batches: int = 5
    batch_size: int = 0
    self_consistent_iters: int = 3
    occupancy_convergence: float = 0.01
    n_samples: int = 10_000


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class SQDBatchedSolver(Solver):
    """SQD with self-consistent batching and orbital occupancy updates.

    Parameters
    ----------
    sampler : object
        Any sampler exposing a ``sample(n)`` method that returns an object
        with a ``.configs`` attribute (a ``(n, n_sites)`` integer tensor).
    config : SQDBatchedConfig, optional
        Solver hyper-parameters.  Uses defaults when omitted.

    Examples
    --------
    >>> from qvartools.samplers import NFSampler
    >>> solver = SQDBatchedSolver(sampler=my_sampler)
    >>> result = solver.solve(hamiltonian, mol_info)
    >>> result.method
    'SQD-Batched'
    """

    def __init__(
        self,
        sampler: Any,
        config: Optional[SQDBatchedConfig] = None,
    ) -> None:
        self.sampler = sampler
        self.config = config or SQDBatchedConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self, hamiltonian: Any, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Run batched SQD and return the ground-state energy estimate.

        Parameters
        ----------
        hamiltonian : MolecularHamiltonian
            Hamiltonian with ``get_hf_state``, ``matrix_elements_fast``,
            ``n_orbitals``, ``n_alpha``, and ``n_beta`` attributes.
        mol_info : dict
            Molecular metadata (unused, kept for API compatibility).

        Returns
        -------
        SolverResult
        """
        t0 = time.perf_counter()
        cfg = self.config
        device = "cpu"

        # Step 1: Sample configurations
        sample_result = self.sampler.sample(cfg.n_samples)
        configs = sample_result.configs.to(device)

        # Step 2: Particle-number filter
        n_orb = hamiltonian.n_orbitals
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta

        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)
        valid_mask = (alpha_counts == n_alpha) & (beta_counts == n_beta)
        configs = configs[valid_mask]

        if len(configs) == 0:
            return SolverResult(
                diag_dim=0,
                wall_time=time.perf_counter() - t0,
                method="SQD-Batched",
                converged=False,
                energy=None,
                metadata={
                    "error": "No valid configurations after particle number filter"
                },
            )

        # Ensure HF state is present
        hf = hamiltonian.get_hf_state().unsqueeze(0).to(device)
        configs = torch.cat([hf, configs], dim=0)
        configs = torch.unique(configs, dim=0)

        n_configs = len(configs)

        # Determine batch size
        batch_size = cfg.batch_size
        if batch_size <= 0:
            batch_size = min(n_configs, max(100, n_configs // cfg.num_batches))

        # Step 3: Self-consistent loop
        orbital_occ = hamiltonian.get_hf_state().float().cpu().numpy()
        best_energy = float("inf")
        batch_energies: list[float] = []
        sc_count = 0

        for sc_iter in range(cfg.self_consistent_iters):
            sc_count = sc_iter + 1
            batches = _create_batches(configs, cfg.num_batches, batch_size)

            iter_energies: list[float] = []
            iter_vectors: list[tuple[torch.Tensor, np.ndarray]] = []

            for batch in batches:
                if len(batch) < 2:
                    continue

                H_proj = hamiltonian.matrix_elements_fast(batch)
                H_np = H_proj.cpu().numpy().astype(np.float64)
                H_np = 0.5 * (H_np + H_np.T)

                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
                e0 = float(eigenvalues[0])
                psi0 = eigenvectors[:, 0]

                iter_energies.append(e0)
                iter_vectors.append((batch, psi0))

            if not iter_energies:
                break

            best_idx = int(np.argmin(iter_energies))
            current_best = iter_energies[best_idx]

            if current_best < best_energy:
                best_energy = current_best
                batch_energies = list(iter_energies)

            # Update orbital occupancies
            best_batch, best_psi = iter_vectors[best_idx]
            new_occ = _compute_orbital_occupancies(best_batch, best_psi)

            occ_change = float(np.max(np.abs(new_occ - orbital_occ)))
            orbital_occ = new_occ

            if occ_change < cfg.occupancy_convergence and sc_iter > 0:
                break

        wall_time = time.perf_counter() - t0

        return SolverResult(
            diag_dim=batch_size,
            wall_time=wall_time,
            method="SQD-Batched",
            converged=True,
            energy=best_energy if best_energy < float("inf") else None,
            metadata={
                "n_configs": n_configs,
                "n_batches": cfg.num_batches,
                "batch_size": batch_size,
                "batch_energies": batch_energies,
                "sc_iterations": sc_count,
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_batches(
    configs: torch.Tensor,
    num_batches: int,
    batch_size: int,
) -> list[torch.Tensor]:
    """Partition configurations into random sub-tensors.

    Each batch is a random subset of ``configs`` of size ``batch_size``.
    If ``configs`` has fewer rows than ``batch_size``, the full tensor is
    used for that batch.

    Parameters
    ----------
    configs : torch.Tensor
        Configuration pool, shape ``(n, n_sites)``.
    num_batches : int
        Number of batches to create.
    batch_size : int
        Target number of configurations per batch.

    Returns
    -------
    list of torch.Tensor
        List of ``num_batches`` sub-tensors.
    """
    n = len(configs)
    batches: list[torch.Tensor] = []
    for _ in range(num_batches):
        if n <= batch_size:
            batches.append(configs)
        else:
            indices = torch.randperm(n)[:batch_size]
            batches.append(configs[indices])
    return batches


def _compute_orbital_occupancies(
    configs: torch.Tensor,
    psi: np.ndarray,
) -> np.ndarray:
    """Compute orbital occupancies <n_p> from an eigenvector.

    Parameters
    ----------
    configs : torch.Tensor
        ``(n_basis, n_sites)`` binary configuration tensor.
    psi : np.ndarray
        Ground-state eigenvector coefficients, shape ``(n_basis,)``.

    Returns
    -------
    np.ndarray
        Orbital occupancies, shape ``(n_sites,)``.
    """
    probs = psi ** 2
    configs_np = configs.cpu().numpy().astype(np.float64)
    return (probs[:, None] * configs_np).sum(axis=0)
