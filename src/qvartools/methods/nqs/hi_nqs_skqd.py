"""
hi_nqs_skqd --- HI+NQS+SKQD: NQS sampling + Krylov expansion + GPU diag
==========================================================================

Iterative pipeline that trains an autoregressive transformer NQS,
samples configurations, expands the basis via Hamiltonian connections
(Krylov-style), and diagonalises in the enlarged subspace.  The
eigenvector is fed back as a teacher signal for the next iteration.

This is the Krylov-augmented variant of :mod:`hi_nqs_sqd`: after NQS
sampling, the basis is expanded by following off-diagonal Hamiltonian
connections from the lowest-energy reference configurations.

External dependencies (``qiskit_addon_sqd``) are optional.

Functions
---------
run_hi_nqs_skqd
    Execute the full HI+NQS+SKQD pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from qvartools._utils.formatting.bitstring_format import (
    configs_to_ibm_format,
    vectorized_dedup,
)
from qvartools._utils.gpu.diagnostics import gpu_solve_fermion
from qvartools.krylov.expansion.krylov_expand import expand_basis_via_connections
from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer
from qvartools.solvers.solver import SolverResult

__all__ = [
    "HINQSSKQDConfig",
    "run_hi_nqs_skqd",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from qiskit_addon_sqd.configuration_recovery import recover_configurations  # type: ignore[import-untyped]
    from qiskit_addon_sqd.fermion import solve_fermion as ibm_solve_fermion  # type: ignore[import-untyped]

    _IBM_SQD_AVAILABLE = True
except ImportError:
    recover_configurations = None  # type: ignore[assignment]
    ibm_solve_fermion = None  # type: ignore[assignment]
    _IBM_SQD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HINQSSKQDConfig:
    """Configuration for the HI+NQS+SKQD pipeline.

    Parameters
    ----------
    n_iterations : int
        Outer self-consistent iterations (default ``10``).
    n_samples_per_iter : int
        NQS samples per iteration (default ``10_000``).
    n_batches : int
        Batches per iteration (default ``5``).
    max_configs_per_batch : int
        Max configs per batch (default ``5000``).
    energy_tol : float
        Convergence threshold in Hartree (default ``1e-5``).
    nqs_lr : float
        NQS learning rate (default ``1e-3``).
    nqs_train_epochs : int
        NQS epochs per iteration (default ``50``).
    embed_dim : int
        Transformer embedding dim (default ``64``).
    n_heads : int
        Attention heads (default ``4``).
    n_layers : int
        Transformer layers per channel (default ``4``).
    temperature : float
        NQS sampling temperature (default ``1.0``).
    krylov_max_new : int
        Max new configs from Krylov expansion per iteration
        (default ``500``).
    krylov_n_ref : int
        Number of reference configs for Krylov expansion
        (default ``10``).
    use_ibm_solver : bool
        Use IBM solver when available (default ``True``).
    device : str
        Torch device (default ``"cpu"``).
    """

    n_iterations: int = 10
    n_samples_per_iter: int = 10_000
    n_batches: int = 5
    max_configs_per_batch: int = 5000
    energy_tol: float = 1e-5
    nqs_lr: float = 1e-3
    nqs_train_epochs: int = 50
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 4
    temperature: float = 1.0
    krylov_max_new: int = 500
    krylov_n_ref: int = 10
    use_ibm_solver: bool = True
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _train_nqs_teacher(
    nqs: AutoregressiveTransformer,
    configs: torch.Tensor,
    coeffs: np.ndarray,
    n_orb: int,
    lr: float,
    epochs: int,
) -> List[float]:
    """Train NQS with eigenvector teacher signal.

    Parameters
    ----------
    nqs : AutoregressiveTransformer
        Transformer NQS.
    configs : torch.Tensor
        Basis configs, shape ``(n_basis, 2*n_orb)``.
    coeffs : np.ndarray
        Eigenvector coefficients, shape ``(n_basis,)``.
    n_orb : int
        Spatial orbitals per spin channel.
    lr : float
        Learning rate.
    epochs : int
        Training epochs.

    Returns
    -------
    list of float
        Per-epoch losses.
    """
    device = next(nqs.parameters()).device

    weights = np.abs(coeffs) ** 2
    total = weights.sum()
    if total > 0:
        weights = weights / total
    weights_t = torch.from_numpy(weights).float().to(device)

    configs_dev = configs.to(device).long()
    alpha = configs_dev[:, :n_orb]
    beta = configs_dev[:, n_orb:]

    optimiser = torch.optim.Adam(nqs.parameters(), lr=lr)
    losses: List[float] = []

    nqs.train()
    for _epoch in range(epochs):
        optimiser.zero_grad()
        log_probs = nqs.log_prob(alpha, beta)
        loss = -(weights_t * log_probs).sum()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.item()))

    nqs.eval()
    return losses


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_hi_nqs_skqd(
    hamiltonian: Any,
    mol_info: Dict[str, Any],
    config: Optional[HINQSSKQDConfig] = None,
) -> SolverResult:
    """Execute the HI+NQS+SKQD pipeline.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian (must support ``diagonal_element`` and
        ``get_connections`` for Krylov expansion).
    mol_info : dict
        Molecular metadata.  Required keys: ``"n_orbitals"``,
        ``"n_alpha"``, ``"n_beta"``, ``"n_qubits"``.
    config : HINQSSKQDConfig or None
        Pipeline configuration.

    Returns
    -------
    SolverResult
        Energy, timing, convergence, and iteration metadata.
    """
    cfg = config or HINQSSKQDConfig()

    n_orb: int = mol_info["n_orbitals"]
    n_alpha: int = mol_info["n_alpha"]
    n_beta: int = mol_info["n_beta"]
    n_qubits: int = mol_info["n_qubits"]
    device = torch.device(cfg.device)

    logger.info(
        "run_hi_nqs_skqd: %d orbitals, %d alpha, %d beta",
        n_orb, n_alpha, n_beta,
    )

    t_start = time.perf_counter()

    # --- Build NQS ---
    nqs = AutoregressiveTransformer(
        n_orbitals=n_orb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    ).to(device)
    nqs.eval()

    # --- Occupancies (uniform prior) ---
    occ_alpha = np.full(n_orb, n_alpha / n_orb)
    occ_beta = np.full(n_orb, n_beta / n_orb)

    # --- Cumulative basis ---
    cumulative_basis = torch.zeros(0, n_qubits, dtype=torch.long, device=device)

    energy_history: List[float] = []
    basis_size_history: List[int] = []
    best_energy = float("inf")
    converged = False

    for iteration in range(cfg.n_iterations):
        logger.info(
            "HI+NQS+SKQD iteration %d / %d", iteration + 1, cfg.n_iterations
        )

        # --- NQS sampling ---
        with torch.no_grad():
            new_configs = nqs.sample(
                cfg.n_samples_per_iter, temperature=cfg.temperature
            ).to(device)

        # Deduplicate (vectorized_dedup expects numpy arrays)
        if cumulative_basis.shape[0] > 0:
            cb_np = cumulative_basis.cpu().numpy()
            nc_np = new_configs.cpu().numpy()
            unique_np = vectorized_dedup(cb_np, nc_np)
            unique_new = torch.from_numpy(unique_np).long().to(device)
        else:
            unique_new = torch.unique(new_configs, dim=0)

        cumulative_basis = torch.cat(
            [cumulative_basis.to(device), unique_new.to(device)], dim=0
        )
        cumulative_basis = torch.unique(cumulative_basis, dim=0)

        # --- Krylov expansion (runs on CPU, move back to device) ---
        pre_expand = cumulative_basis.shape[0]
        cumulative_basis = expand_basis_via_connections(
            cumulative_basis,
            hamiltonian,
            max_new=cfg.krylov_max_new,
            n_ref=cfg.krylov_n_ref,
        )
        cumulative_basis = cumulative_basis.to(device)
        n_krylov_added = cumulative_basis.shape[0] - pre_expand

        logger.info(
            "  NQS: %d unique new, Krylov: +%d, total basis: %d",
            unique_new.shape[0],
            n_krylov_added,
            cumulative_basis.shape[0],
        )

        # --- Batch diagonalisation ---
        batch_energies: List[float] = []
        best_coeffs: Optional[np.ndarray] = None
        best_batch_configs: Optional[torch.Tensor] = None
        best_batch_energy = float("inf")
        latest_occs: Any = None

        for _batch_idx in range(cfg.n_batches):
            if cumulative_basis.shape[0] > cfg.max_configs_per_batch:
                indices = torch.randperm(cumulative_basis.shape[0])[
                    : cfg.max_configs_per_batch
                ]
                batch_configs = cumulative_basis[indices]
            else:
                batch_configs = cumulative_basis

            if _IBM_SQD_AVAILABLE and cfg.use_ibm_solver:
                ibm_data = configs_to_ibm_format(batch_configs, n_orb, n_qubits)
                recovered = recover_configurations(
                    ibm_data,
                    (n_alpha, n_beta),
                    (occ_alpha, occ_beta),
                )
                e_b, coeffs_b, occs_b = ibm_solve_fermion(
                    recovered, (n_alpha, n_beta), mol_info
                )
            else:
                e_b, coeffs_b, occs_b = gpu_solve_fermion(
                    batch_configs, hamiltonian
                )

            e_b = float(e_b)
            batch_energies.append(e_b)
            latest_occs = occs_b

            if e_b < best_batch_energy:
                best_batch_energy = e_b
                best_coeffs = np.asarray(coeffs_b)
                best_batch_configs = batch_configs

        iter_energy = float(np.min(batch_energies))
        energy_history.append(iter_energy)
        basis_size_history.append(int(cumulative_basis.shape[0]))
        best_energy = min(best_energy, iter_energy)

        # --- Update occupancies ---
        if isinstance(latest_occs, tuple) and len(latest_occs) == 2:
            occ_alpha = np.clip(
                np.asarray(latest_occs[0], dtype=np.float64), 0.0, 1.0
            )
            occ_beta = np.clip(
                np.asarray(latest_occs[1], dtype=np.float64), 0.0, 1.0
            )

        # --- NQS teacher training ---
        if best_coeffs is not None and best_batch_configs is not None:
            _train_nqs_teacher(
                nqs,
                best_batch_configs,
                best_coeffs,
                n_orb,
                lr=cfg.nqs_lr,
                epochs=cfg.nqs_train_epochs,
            )

        logger.info(
            "  energy=%.8f best=%.8f basis=%d",
            iter_energy,
            best_energy,
            cumulative_basis.shape[0],
        )

        # --- Convergence ---
        if len(energy_history) >= 2:
            delta = abs(energy_history[-1] - energy_history[-2])
            if delta < cfg.energy_tol:
                converged = True
                logger.info("  converged: |dE|=%.2e", delta)
                break

    wall_time = time.perf_counter() - t_start

    return SolverResult(
        energy=best_energy,
        diag_dim=int(cumulative_basis.shape[0]),
        wall_time=wall_time,
        method="HI+NQS+SKQD",
        converged=converged,
        metadata={
            "energy_history": energy_history,
            "basis_size_history": basis_size_history,
            "n_iterations": len(energy_history),
            "final_basis_size": int(cumulative_basis.shape[0]),
        },
    )
