"""
cipsi --- CIPSI Selected-CI solver
===================================

Implements :class:`CIPSISolver`, a Configuration Interaction using a
Perturbative Selection made Iteratively (CIPSI) solver that builds a
compact CI basis by selecting the most important determinants via
second-order perturbation theory.

Algorithm
---------
1. Seed the basis with the Hartree--Fock determinant.
2. At each iteration:
   a. Build the projected Hamiltonian and diagonalize.
   b. For every basis determinant, enumerate single/double excitations
      (H-connections) not already in the basis.
   c. Score each candidate *x* by PT2 importance:
      ``eps_x = |<x|H|Phi>|^2 / |E_0 - H_xx|``
   d. Add the top-k highest-scoring candidates.
   e. Stop when |dE| < threshold or the basis saturates.
3. Return the variational ground-state energy and basis size.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch

from qvartools._utils.hashing.config_hash import config_integer_hash
from qvartools.solvers.solver import Solver, SolverResult

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

__all__ = [
    "CIPSISolver",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
_MAX_ITERATIONS = 30
_MAX_BASIS_SIZE = 10_000
_CONVERGENCE_THRESHOLD = 1e-5  # Hartree
_EXPANSION_SIZE = 500


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class CIPSISolver(Solver):
    """Selected CI solver using the CIPSI perturbative selection scheme.

    Parameters
    ----------
    max_iterations : int
        Maximum number of expansion iterations.
    max_basis_size : int
        Hard cap on basis dimension.
    convergence_threshold : float
        Energy change threshold (Ha) for early stopping.
    expansion_size : int
        Number of candidates added per iteration.
    """

    def __init__(
        self,
        max_iterations: int = _MAX_ITERATIONS,
        max_basis_size: int = _MAX_BASIS_SIZE,
        convergence_threshold: float = _CONVERGENCE_THRESHOLD,
        expansion_size: int = _EXPANSION_SIZE,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_basis_size = max_basis_size
        self.convergence_threshold = convergence_threshold
        self.expansion_size = expansion_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self, hamiltonian: Any, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Run the CIPSI selected-CI algorithm.

        Parameters
        ----------
        hamiltonian : MolecularHamiltonian
            Hamiltonian exposing ``get_hf_state()``, ``matrix_elements_fast()``,
            ``get_connections()``, ``diagonal_element()``, and
            ``diagonal_elements_batch()``.
        mol_info : dict
            Molecular metadata (unused, kept for API compatibility).

        Returns
        -------
        SolverResult
        """
        t0 = time.perf_counter()

        # 1. Seed with the HF determinant
        hf_config = hamiltonian.get_hf_state()
        if hf_config.dim() == 1:
            hf_config = hf_config.unsqueeze(0)

        basis = hf_config.clone()
        basis_hashes: set[int] = set(config_integer_hash(basis))

        prev_energy: float | None = None
        converged = False
        iteration_energies: list[float] = []
        e0 = float("nan")

        iterator: Any = range(self.max_iterations)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="CIPSI", leave=False)

        # 2. Main loop
        for it in iterator:
            n_basis = basis.shape[0]

            # (a) Build and diagonalize projected H
            h_matrix = hamiltonian.matrix_elements_fast(basis)
            h_np = np.asarray(h_matrix, dtype=np.float64)
            h_np = 0.5 * (h_np + h_np.T)
            eigvals, eigvecs = np.linalg.eigh(h_np)
            e0 = float(eigvals[0])
            coeffs = eigvecs[:, 0]

            iteration_energies.append(e0)
            logger.debug(
                "CIPSI iter %d: basis=%d  E=%.10f Ha", it, n_basis, e0
            )

            # Convergence check
            if prev_energy is not None:
                delta_e = abs(e0 - prev_energy)
                if delta_e < self.convergence_threshold:
                    converged = True
                    logger.info(
                        "CIPSI converged at iter %d: |dE|=%.2e < %.2e",
                        it,
                        delta_e,
                        self.convergence_threshold,
                    )
                    break
            prev_energy = e0

            # Basis-size guard
            if n_basis >= self.max_basis_size:
                logger.warning(
                    "CIPSI basis reached max size (%d); stopping.",
                    self.max_basis_size,
                )
                break

            # (b-c) Collect candidates and accumulate PT2 numerators
            numerator_accum: dict[int, float] = defaultdict(float)
            candidate_configs: dict[int, torch.Tensor] = {}

            for idx in range(n_basis):
                c_i = float(coeffs[idx])
                if abs(c_i) < 1e-14:
                    continue

                connections, h_elements = hamiltonian.get_connections(
                    basis[idx]
                )
                if connections is None or len(connections) == 0:
                    continue

                conn_hashes = config_integer_hash(connections)
                for j, h_conn in enumerate(conn_hashes):
                    if h_conn in basis_hashes:
                        continue
                    numerator_accum[h_conn] += c_i * float(h_elements[j])
                    if h_conn not in candidate_configs:
                        candidate_configs[h_conn] = connections[j]

            if not candidate_configs:
                logger.info("CIPSI: no new candidates found; stopping.")
                break

            # (d) Compute PT2 importance
            cand_hash_list = list(candidate_configs.keys())
            cand_tensor = torch.stack(
                [candidate_configs[h] for h in cand_hash_list]
            )

            h_diag = np.asarray(
                hamiltonian.diagonal_elements_batch(cand_tensor),
                dtype=np.float64,
            )

            importances = np.empty(len(cand_hash_list), dtype=np.float64)
            for k, h_key in enumerate(cand_hash_list):
                numer_sq = numerator_accum[h_key] ** 2
                denom = abs(e0 - h_diag[k])
                importances[k] = numer_sq / max(denom, 1e-14)

            # (e) Select top-k and expand basis
            room = self.max_basis_size - n_basis
            n_add = min(self.expansion_size, len(cand_hash_list), room)
            if n_add >= len(cand_hash_list):
                top_indices = np.arange(len(cand_hash_list))
            else:
                top_indices = np.argpartition(-importances, n_add)[:n_add]

            new_configs = cand_tensor[top_indices]
            new_hashes = [cand_hash_list[i] for i in top_indices]

            basis = torch.cat([basis, new_configs], dim=0)
            basis_hashes.update(new_hashes)

            if tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    E=f"{e0:.8f}", basis=basis.shape[0], ordered=False
                )

        # 3. Final result
        wall_time = time.perf_counter() - t0
        diag_dim = basis.shape[0]

        logger.info(
            "CIPSI finished: E=%.10f Ha  basis=%d  converged=%s  time=%.2fs",
            e0,
            diag_dim,
            converged,
            wall_time,
        )

        return SolverResult(
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="CIPSI",
            converged=converged,
            energy=e0,
            metadata={
                "n_iterations": len(iteration_energies),
                "iteration_energies": iteration_energies,
                "max_iterations": self.max_iterations,
                "expansion_size": self.expansion_size,
                "convergence_threshold": self.convergence_threshold,
            },
        )
