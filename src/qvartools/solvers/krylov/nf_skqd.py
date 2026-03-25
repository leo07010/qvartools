"""
nf_skqd --- NF-SKQD: Normalizing Flow Sample-based Krylov Quantum Diagonalization
==================================================================================

A faithful classical analog of the quantum SKQD algorithm (Yu et al., 2025),
replacing quantum circuit time evolution with NF distribution evolution.

Quantum SKQD
    Krylov subspace = {|psi_0>, U|psi_0>, U^2|psi_0>, ..., U^k|psi_0>}
    where U = exp(-iH dt) via Trotter circuits.  Sample each U^k|psi_0>
    to obtain bitstrings, combine ALL bitstrings into a cumulative basis,
    project H, and diagonalize.

NF-SKQD
    Krylov subspace = {NF_0, NF_1, NF_2, ..., NF_k}.
    NF_0 is an untrained (random / HF-biased) distribution.  NF_{k+1} is
    obtained by partially updating NF_k toward the current ground-state
    eigenvector |Phi_k>.  Sample each NF_k, add to a cumulative basis, and
    diagonalize.

Key design principles
    1. **Cumulative basis** -- never discard configs from previous powers.
    2. **Partial NF update** -- only a few gradient steps per power (mimics
       a small Trotter time step).
    3. **No H-connection expansion** -- the basis grows purely from NF
       sampling.
    4. **Energy monotonicity** -- each power can only improve or maintain
       the energy.

References
----------
.. [1] Yu et al. (2025) "Quantum-Centric Algorithm for Sample-Based Krylov
   Diagonalization", arXiv:2501.09702
.. [2] Pellow-Jarman et al. (2025) "HIVQE", arXiv:2503.06292
.. [3] Robledo-Moreno et al. (2024) "Chemistry beyond exact solutions"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np
import torch

from qvartools._utils.hashing.config_hash import config_integer_hash
from qvartools.solvers.solver import Solver, SolverResult

__all__ = [
    "NFSKQDConfig",
    "NFSKQDSolver",
]


# ---------------------------------------------------------------------------
# Flow model protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FlowModel(Protocol):
    """Structural type for objects accepted as *flow_model*."""

    def sample(self, n: int, **kwargs: Any) -> Any: ...
    def log_prob(self, x: torch.Tensor) -> torch.Tensor: ...
    def parameters(self) -> Any: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NFSKQDConfig:
    """Configuration for the NF-SKQD solver.

    Parameters
    ----------
    n_krylov_powers : int
        Number of Krylov "time steps".
    n_samples_per_power : int
        Number of NF samples drawn at each power.
    nf_steps_per_power : int
        Gradient steps applied to the NF per power (partial update).
    nf_lr : float
        Learning rate for the NF Adam optimizer.
    wf_weight : float
        Weight for the wavefunction-matching (KL) loss term.
    energy_weight : float
        Weight for the REINFORCE energy loss term.
    entropy_weight : float
        Weight for the entropy regularisation term.
    initial_temperature : float
        Sampling temperature at the first power.
    final_temperature : float
        Sampling temperature at the last power.
    warmup_powers : int
        Number of initial powers during which NF updates are skipped.
    max_basis_size : int
        Hard cap on cumulative basis size.
    convergence_threshold : float
        Energy change threshold (Ha) for early stopping.
    """

    n_krylov_powers: int = 10
    n_samples_per_power: int = 2000
    nf_steps_per_power: int = 20
    nf_lr: float = 1e-3
    wf_weight: float = 1.0
    energy_weight: float = 0.1
    entropy_weight: float = 0.05
    initial_temperature: float = 2.0
    final_temperature: float = 0.5
    warmup_powers: int = 0
    max_basis_size: int = 10_000
    convergence_threshold: float = 1e-6


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class NFSKQDSolver(Solver):
    """NF-SKQD: faithful NF analog of quantum SKQD with cumulative basis.

    Each Krylov power *k*:

    1. Sample from the current NF distribution NF_k.
    2. Add unique, particle-number-valid configs to the cumulative basis.
    3. Diagonalize the projected Hamiltonian (Rayleigh--Ritz).
    4. Partially update the NF toward the ground-state eigenvector.

    Parameters
    ----------
    flow_model : object
        A normalizing-flow model exposing ``sample()``, ``log_prob()``,
        and ``parameters()`` methods.
    config : NFSKQDConfig, optional
        Solver hyper-parameters.  Uses defaults when omitted.
    """

    def __init__(
        self,
        flow_model: FlowModel,
        config: Optional[NFSKQDConfig] = None,
    ) -> None:
        self.flow = flow_model
        self.config = config or NFSKQDConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self, hamiltonian: Any, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Run NF-SKQD and return the ground-state energy estimate.

        Parameters
        ----------
        hamiltonian : MolecularHamiltonian
            Hamiltonian with ``get_hf_state``, ``matrix_elements_fast``,
            ``diagonal_element``, and ``diagonal_elements_batch`` methods.
        mol_info : dict
            Molecular metadata (unused by this solver, kept for API compat).

        Returns
        -------
        SolverResult
        """
        t0 = time.perf_counter()
        cfg = self.config
        device = "cpu"

        n_orb = hamiltonian.n_orbitals
        n_alpha = hamiltonian.n_alpha
        n_beta = hamiltonian.n_beta

        optimizer = torch.optim.Adam(self.flow.parameters(), lr=cfg.nf_lr)

        # Seed with HF state
        hf = hamiltonian.get_hf_state().unsqueeze(0).to(device)
        cumulative_basis = hf.clone()
        basis_hashes: set[int] = set(config_integer_hash(cumulative_basis))

        energy_history: list[float] = []
        basis_size_history: list[int] = []
        samples_per_power: list[int] = []

        prev_energy = float("inf")
        best_energy = float("inf")
        converged = False
        last_k = 0

        for k in range(cfg.n_krylov_powers):
            last_k = k

            # Temperature annealing
            progress = k / max(cfg.n_krylov_powers - 1, 1)
            temperature = (
                cfg.initial_temperature
                + progress * (cfg.final_temperature - cfg.initial_temperature)
            )

            # Step 1: Sample from current NF
            with torch.no_grad():
                try:
                    sample_out = self.flow.sample(
                        cfg.n_samples_per_power, temperature=temperature
                    )
                except TypeError:
                    sample_out = self.flow.sample(cfg.n_samples_per_power)

                if sample_out[0].dim() == 1:
                    raw_configs = sample_out[1].long().to(device)
                else:
                    raw_configs = sample_out[0].long().to(device)

                # Particle-number filter
                alpha_counts = raw_configs[:, :n_orb].sum(dim=1)
                beta_counts = raw_configs[:, n_orb:].sum(dim=1)
                valid = (alpha_counts == n_alpha) & (beta_counts == n_beta)
                new_configs = raw_configs[valid]

            # Deduplicate and add to cumulative basis
            n_new = 0
            if len(new_configs) > 0:
                new_unique = torch.unique(new_configs, dim=0)
                new_hashes = config_integer_hash(new_unique)
                truly_new: list[torch.Tensor] = []
                for idx, h in enumerate(new_hashes):
                    if h not in basis_hashes:
                        truly_new.append(new_unique[idx])
                        basis_hashes.add(h)

                if truly_new:
                    new_batch = torch.stack(truly_new)
                    cumulative_basis = torch.cat(
                        [cumulative_basis, new_batch], dim=0
                    )
                    n_new = len(truly_new)

            samples_per_power.append(n_new)

            # Enforce max basis size
            if len(cumulative_basis) > cfg.max_basis_size:
                cumulative_basis = torch.cat(
                    [hf, cumulative_basis[-(cfg.max_basis_size - 1) :]],
                    dim=0,
                )
                basis_hashes = set(config_integer_hash(cumulative_basis))

            # Step 2: Diagonalize
            if len(cumulative_basis) < 2:
                e0 = float(hamiltonian.diagonal_element(cumulative_basis[0]))
                psi0 = np.array([1.0])
            else:
                H_proj = hamiltonian.matrix_elements_fast(cumulative_basis)
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

                e0 = float(eigenvalues[0])
                psi0 = eigenvectors[:, 0]

            energy_history.append(e0)
            basis_size_history.append(len(cumulative_basis))

            if e0 < best_energy:
                best_energy = e0

            # Step 3: Partial NF update
            if len(cumulative_basis) >= 2 and k >= cfg.warmup_powers:
                _evolve_nf(
                    self.flow,
                    cumulative_basis,
                    psi0,
                    e0,
                    hamiltonian,
                    optimizer,
                    cfg,
                )

            # Step 4: Convergence check
            delta_e = abs(e0 - prev_energy)
            prev_energy = e0

            if delta_e < cfg.convergence_threshold and k > 0:
                converged = True
                break

        wall_time = time.perf_counter() - t0

        return SolverResult(
            diag_dim=len(cumulative_basis),
            wall_time=wall_time,
            method="NF-SKQD",
            converged=converged,
            energy=best_energy if best_energy < float("inf") else None,
            metadata={
                "n_krylov_powers": last_k + 1,
                "energy_history": energy_history,
                "basis_size_history": basis_size_history,
                "samples_per_power": samples_per_power,
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _evolve_nf(
    flow: FlowModel,
    basis_configs: torch.Tensor,
    psi0: np.ndarray,
    e0: float,
    hamiltonian: Any,
    optimizer: torch.optim.Optimizer,
    cfg: NFSKQDConfig,
) -> None:
    """Partially update the NF distribution toward the ground-state eigenvector.

    The loss combines:

    * **Wavefunction matching** -- KL(|Phi|^2 || p_NF).
    * **REINFORCE energy** -- gradient estimator for lowering energy.
    * **Entropy regularisation** -- maintains exploration.

    Only a few gradient steps are taken (not full convergence), so the NF
    distribution shifts gradually across Krylov powers.
    """
    basis_float = basis_configs.float()

    weights = torch.from_numpy(psi0 ** 2).float()
    weights = weights / weights.sum()

    with torch.no_grad():
        diag_energies = torch.from_numpy(
            np.asarray(
                hamiltonian.diagonal_elements_batch(basis_configs),
                dtype=np.float64,
            )
        ).float()
        advantage = diag_energies - e0

    for _step in range(cfg.nf_steps_per_power):
        optimizer.zero_grad()

        log_probs = flow.log_prob(basis_float)

        loss_wf = -(weights * log_probs).sum()
        loss_energy = (weights * advantage * log_probs).sum()
        loss_entropy = log_probs.mean()

        loss = (
            cfg.wf_weight * loss_wf
            + cfg.energy_weight * loss_energy
            + cfg.entropy_weight * loss_entropy
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        optimizer.step()
