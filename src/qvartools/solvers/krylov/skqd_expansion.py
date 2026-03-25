"""
skqd_expansion --- SKQD Methods B and C (H-connection expansion solvers)
=========================================================================

Implements two SKQD expansion strategies that start from NF-sampled
configurations and iteratively grow the basis via Hamiltonian connections:

* :class:`SKQDSolverB` -- PT2 importance scoring for candidate selection.
* :class:`SKQDSolverC` -- coupling strength scoring (|c_i| * |H_{x',x_i}|).

Both solvers accept an external sampler to generate the initial basis,
filter for correct particle number, and then expand deterministically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from qvartools._utils.hashing.config_hash import config_integer_hash
from qvartools.solvers.solver import Solver, SolverResult

__all__ = [
    "SKQDExpansionConfig",
    "SKQDSolverB",
    "SKQDSolverC",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SKQDExpansionConfig:
    """Configuration for SKQD expansion solvers (Methods B and C).

    Parameters
    ----------
    n_samples : int
        Number of configurations drawn from the sampler for the initial basis.
    max_iterations : int
        Maximum number of expansion iterations.
    expansion_size : int
        Number of candidates added per iteration.
    max_basis_size : int
        Hard cap on basis dimension.
    convergence_threshold : float
        Energy change threshold (Ha) for early stopping.
    n_process : int
        Number of high-amplitude configs whose connections are explored
        per iteration.
    """

    n_samples: int = 5000
    max_iterations: int = 20
    expansion_size: int = 500
    max_basis_size: int = 10_000
    convergence_threshold: float = 1e-5
    n_process: int = 200


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_initial_basis(
    sampler: Any,
    hamiltonian: Any,
    n_samples: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, set[int]]:
    """Sample, filter for particle number, add HF, deduplicate.

    Returns ``(basis, basis_hashes)``.
    """
    sample_result = sampler.sample(n_samples)
    basis = sample_result.configs.to(device).long()

    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta

    alpha_counts = basis[:, :n_orb].sum(dim=1)
    beta_counts = basis[:, n_orb:].sum(dim=1)
    valid = (alpha_counts == n_alpha) & (beta_counts == n_beta)
    basis = basis[valid]

    hf = hamiltonian.get_hf_state().unsqueeze(0).to(device)
    basis = torch.cat([hf, basis], dim=0)
    basis = torch.unique(basis, dim=0)

    basis_hashes: set[int] = set(config_integer_hash(basis))
    return basis, basis_hashes


def _diagonalize(
    hamiltonian: Any, basis: torch.Tensor
) -> tuple[float, np.ndarray]:
    """Diagonalize the projected Hamiltonian. Returns ``(e0, psi0)``."""
    H_proj = hamiltonian.matrix_elements_fast(basis)
    H_np = H_proj.cpu().numpy().astype(np.float64)
    H_np = 0.5 * (H_np + H_np.T)

    if len(H_np) <= 2000:
        eigenvalues, eigenvectors = np.linalg.eigh(H_np)
    else:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        eigenvalues, eigenvectors = eigsh(
            csr_matrix(H_np), k=1, which="SA"
        )

    return float(eigenvalues[0]), eigenvectors[:, 0]


# ---------------------------------------------------------------------------
# SKQD Method B
# ---------------------------------------------------------------------------


class SKQDSolverB(Solver):
    """SKQD Method B: H-connection expansion with PT2 importance scoring.

    Pipeline:

    1. Sample initial basis from an external sampler.
    2. Diagonalize the projected Hamiltonian.
    3. For top-amplitude configs, enumerate H-connections outside the basis.
    4. Rank candidates by PT2 importance
       ``|c_i * H_{x', x_i}|^2 / |E_0 - H_{x'x'}|``.
    5. Add top-k, re-diagonalize.
    6. Repeat until convergence.

    Parameters
    ----------
    sampler : object
        Sampler with a ``sample(n)`` method returning an object with a
        ``.configs`` attribute.
    config : SKQDExpansionConfig, optional
        Solver hyper-parameters.
    """

    def __init__(
        self,
        sampler: Any,
        config: Optional[SKQDExpansionConfig] = None,
    ) -> None:
        self.sampler = sampler
        self.config = config or SKQDExpansionConfig()

    def solve(
        self, hamiltonian: Any, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Run SKQD-B.

        Parameters
        ----------
        hamiltonian : MolecularHamiltonian
            Hamiltonian with the standard method set.
        mol_info : dict
            Molecular metadata (unused).

        Returns
        -------
        SolverResult
        """
        t0 = time.perf_counter()
        cfg = self.config

        basis, basis_hashes = _build_initial_basis(
            self.sampler, hamiltonian, cfg.n_samples
        )

        prev_energy = float("inf")
        converged = False
        e0: float | None = None
        delta_e: float | None = None
        iteration_count = 0

        for iteration in range(cfg.max_iterations):
            iteration_count = iteration + 1

            if len(basis) > cfg.max_basis_size:
                break

            e0, psi0 = _diagonalize(hamiltonian, basis)

            delta_e = abs(e0 - prev_energy)
            if delta_e < cfg.convergence_threshold and iteration > 0:
                converged = True
                break
            prev_energy = e0

            # Find new configs via H-connections
            new_configs: list[torch.Tensor] = []
            new_importance: list[float] = []

            sorted_idx = np.argsort(np.abs(psi0))[::-1]
            n_process = min(len(sorted_idx), cfg.n_process)

            for idx in sorted_idx[:n_process]:
                c_i = psi0[idx]
                if abs(c_i) < 1e-8:
                    continue

                connected, elements = hamiltonian.get_connections(basis[idx])
                if connected is None or len(connected) == 0:
                    continue

                conn_hashes = config_integer_hash(connected)
                for k, h in enumerate(conn_hashes):
                    if h not in basis_hashes:
                        h_elem = float(elements[k])
                        coupling = c_i * h_elem
                        h_xx = float(
                            hamiltonian.diagonal_element(connected[k])
                        )
                        denom = abs(e0 - h_xx) + 1e-12
                        importance = coupling ** 2 / denom

                        new_configs.append(connected[k])
                        new_importance.append(importance)
                        basis_hashes.add(h)

            if not new_configs:
                converged = True
                break

            importance_arr = np.array(new_importance)
            top_k = min(cfg.expansion_size, len(new_configs))
            top_indices = np.argsort(importance_arr)[-top_k:]

            new_batch = torch.stack([new_configs[i] for i in top_indices])
            basis = torch.cat([basis, new_batch], dim=0)

        wall_time = time.perf_counter() - t0

        return SolverResult(
            diag_dim=len(basis),
            wall_time=wall_time,
            method="NF-SKQD-B",
            converged=converged,
            energy=e0,
            metadata={
                "iterations": iteration_count,
                "final_basis_size": len(basis),
                "delta_e": delta_e,
            },
        )


# ---------------------------------------------------------------------------
# SKQD Method C
# ---------------------------------------------------------------------------


class SKQDSolverC(Solver):
    """SKQD Method C: H-connection expansion with coupling strength scoring.

    Pipeline:

    1. Sample initial basis from an external sampler.
    2. Diagonalize the projected Hamiltonian.
    3. For top-amplitude ("boundary") configs, enumerate H-connections
       outside the basis.
    4. Rank candidates by coupling score ``|c_i| * |H_{x', x_i}|``.
    5. Add top-k, re-diagonalize.
    6. Repeat until convergence.

    Parameters
    ----------
    sampler : object
        Sampler with a ``sample(n)`` method returning an object with a
        ``.configs`` attribute.
    config : SKQDExpansionConfig, optional
        Solver hyper-parameters.
    """

    def __init__(
        self,
        sampler: Any,
        config: Optional[SKQDExpansionConfig] = None,
    ) -> None:
        self.sampler = sampler
        self.config = config or SKQDExpansionConfig()

    def solve(
        self, hamiltonian: Any, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Run SKQD-C.

        Parameters
        ----------
        hamiltonian : MolecularHamiltonian
            Hamiltonian with the standard method set.
        mol_info : dict
            Molecular metadata (unused).

        Returns
        -------
        SolverResult
        """
        t0 = time.perf_counter()
        cfg = self.config

        basis, basis_hashes = _build_initial_basis(
            self.sampler, hamiltonian, cfg.n_samples
        )

        prev_energy = float("inf")
        converged = False
        e0: float | None = None
        delta_e: float | None = None
        iteration_count = 0

        for iteration in range(cfg.max_iterations):
            iteration_count = iteration + 1

            if len(basis) > cfg.max_basis_size:
                break

            e0, psi0 = _diagonalize(hamiltonian, basis)

            delta_e = abs(e0 - prev_energy)
            if delta_e < cfg.convergence_threshold and iteration > 0:
                converged = True
                break
            prev_energy = e0

            # Identify boundary configs and collect external connections
            sorted_idx = np.argsort(np.abs(psi0))[::-1]
            n_boundary = min(len(sorted_idx), cfg.n_process)

            external_configs: list[torch.Tensor] = []
            external_scores: list[float] = []

            for idx in sorted_idx[:n_boundary]:
                c_i = abs(psi0[idx])
                if c_i < 1e-8:
                    continue

                connected, elements = hamiltonian.get_connections(basis[idx])
                if connected is None or len(connected) == 0:
                    continue

                conn_hashes = config_integer_hash(connected)
                for k, h in enumerate(conn_hashes):
                    if h not in basis_hashes:
                        score = c_i * abs(float(elements[k]))
                        external_configs.append(connected[k])
                        external_scores.append(score)
                        basis_hashes.add(h)

            if not external_configs:
                converged = True
                break

            scores_arr = np.array(external_scores)
            top_k = min(cfg.expansion_size, len(external_configs))
            top_indices = np.argsort(scores_arr)[-top_k:]

            new_batch = torch.stack(
                [external_configs[i] for i in top_indices]
            )
            basis = torch.cat([basis, new_batch], dim=0)

        wall_time = time.perf_counter() - t0

        return SolverResult(
            diag_dim=len(basis),
            wall_time=wall_time,
            method="NF-SKQD-C",
            converged=converged,
            energy=e0,
            metadata={
                "iterations": iteration_count,
                "final_basis_size": len(basis),
                "delta_e": delta_e,
            },
        )
