"""
nqs_skqd --- NQS+SKQD: two-stage (train NQS then SKQD)
========================================================

Two-stage pipeline: train an autoregressive transformer NQS, sample
configurations, expand the basis via Hamiltonian connections, and
diagonalise in the enlarged subspace.

This is a single-pass variant --- no iterative feedback.  For the
iterative version see :mod:`hi_nqs_skqd`.

Functions
---------
run_nqs_skqd
    Execute the NQS+SKQD pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from qvartools._utils.gpu.diagnostics import gpu_solve_fermion
from qvartools.krylov.expansion.krylov_expand import expand_basis_via_connections
from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer
from qvartools.solvers.solver import SolverResult

__all__ = [
    "NQSSKQDConfig",
    "run_nqs_skqd",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NQSSKQDConfig:
    """Configuration for the NQS+SKQD pipeline.

    Parameters
    ----------
    n_samples : int
        NQS samples to draw (default ``10_000``).
    nqs_train_epochs : int
        VMC pre-training epochs (default ``200``).
    nqs_lr : float
        NQS learning rate (default ``1e-3``).
    embed_dim : int
        Transformer embedding dim (default ``64``).
    n_heads : int
        Attention heads (default ``4``).
    n_layers : int
        Transformer layers per channel (default ``4``).
    temperature : float
        Sampling temperature (default ``1.0``).
    krylov_max_new : int
        Max new configs from Krylov expansion (default ``500``).
    krylov_n_ref : int
        Reference configs for expansion (default ``10``).
    max_basis_size : int
        Max basis size for diag (default ``10_000``).
    device : str
        Torch device (default ``"cpu"``).
    """

    n_samples: int = 10_000
    nqs_train_epochs: int = 200
    nqs_lr: float = 1e-3
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 4
    temperature: float = 1.0
    krylov_max_new: int = 500
    krylov_n_ref: int = 10
    max_basis_size: int = 10_000
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_nqs_skqd(
    hamiltonian: Any,
    mol_info: Dict[str, Any],
    config: Optional[NQSSKQDConfig] = None,
) -> SolverResult:
    """Execute the NQS+SKQD pipeline.

    Stage 1 --- Train an autoregressive transformer NQS via VMC.

    Stage 2 --- Sample, deduplicate, expand via Hamiltonian connections,
    and diagonalise.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian (must support ``diagonal_element`` and
        ``get_connections``).
    mol_info : dict
        Molecular metadata.  Required keys: ``"n_orbitals"``,
        ``"n_alpha"``, ``"n_beta"``, ``"n_qubits"``.
    config : NQSSKQDConfig or None
        Pipeline configuration.

    Returns
    -------
    SolverResult
        Energy, basis dimension, wall time, and metadata.
    """
    cfg = config or NQSSKQDConfig()

    n_orb: int = mol_info["n_orbitals"]
    n_alpha: int = mol_info["n_alpha"]
    n_beta: int = mol_info["n_beta"]
    n_qubits: int = mol_info["n_qubits"]
    device = torch.device(cfg.device)

    logger.info(
        "run_nqs_skqd: %d orbitals, %d alpha, %d beta", n_orb, n_alpha, n_beta
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

    # --- Stage 1: VMC pre-training ---
    optimiser = torch.optim.Adam(nqs.parameters(), lr=cfg.nqs_lr)
    train_energies: List[float] = []

    nqs.train()
    for _epoch in range(cfg.nqs_train_epochs):
        optimiser.zero_grad()

        with torch.no_grad():
            configs = nqs.sample(256, temperature=cfg.temperature).to(device)

        alpha = configs[:, :n_orb]
        beta = configs[:, n_orb:]
        log_probs = nqs.log_prob(alpha, beta)

        local_energies = torch.zeros(configs.shape[0], device=device)
        for i in range(configs.shape[0]):
            local_energies[i] = hamiltonian.diagonal_element(configs[i])

        baseline = local_energies.mean()
        loss = ((local_energies - baseline) * log_probs).mean()
        loss.backward()
        optimiser.step()

        train_energies.append(float(baseline.item()))

    nqs.eval()
    logger.info("  NQS training done, final E ~ %.6f", train_energies[-1])

    # --- Stage 2: Sample, expand, diagonalise ---
    with torch.no_grad():
        sampled = nqs.sample(cfg.n_samples, temperature=cfg.temperature).to(device)

    basis = torch.unique(sampled, dim=0)
    pre_expand = basis.shape[0]

    # Krylov expansion
    basis = expand_basis_via_connections(
        basis,
        hamiltonian,
        max_new=cfg.krylov_max_new,
        n_ref=cfg.krylov_n_ref,
    )
    n_krylov_added = basis.shape[0] - pre_expand

    if basis.shape[0] > cfg.max_basis_size:
        basis = basis[: cfg.max_basis_size]

    logger.info(
        "  basis: %d NQS + %d Krylov = %d total",
        pre_expand, n_krylov_added, basis.shape[0],
    )

    energy, eigvec, occs = gpu_solve_fermion(basis, hamiltonian)

    wall_time = time.perf_counter() - t_start

    return SolverResult(
        energy=float(energy),
        diag_dim=int(basis.shape[0]),
        wall_time=wall_time,
        method="NQS+SKQD",
        converged=True,
        metadata={
            "train_energies": train_energies,
            "n_nqs_configs": pre_expand,
            "n_krylov_added": n_krylov_added,
            "n_unique_configs": int(basis.shape[0]),
        },
    )
