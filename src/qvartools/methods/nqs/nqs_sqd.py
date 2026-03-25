"""
nqs_sqd --- NQS+SQD: two-stage (train NQS then SQD)
=====================================================

Simple two-stage pipeline: first train an autoregressive transformer NQS
to learn the ground-state distribution, then sample configurations and
diagonalise the projected Hamiltonian in the sampled subspace.

Unlike the iterative ``HI+NQS+SQD`` variant, this method performs a
single pass with no feedback loop.

Functions
---------
run_nqs_sqd
    Execute the NQS+SQD pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from qvartools._utils.gpu.diagnostics import gpu_solve_fermion
from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer
from qvartools.solvers.solver import SolverResult

__all__ = [
    "NQSSQDConfig",
    "run_nqs_sqd",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NQSSQDConfig:
    """Configuration for the NQS+SQD pipeline.

    Parameters
    ----------
    n_samples : int
        Number of NQS samples to draw (default ``10_000``).
    nqs_train_epochs : int
        Pre-training epochs for NQS (default ``200``).
    nqs_lr : float
        NQS learning rate (default ``1e-3``).
    embed_dim : int
        Transformer embedding dimension (default ``64``).
    n_heads : int
        Attention heads (default ``4``).
    n_layers : int
        Transformer layers per channel (default ``4``).
    temperature : float
        Sampling temperature (default ``1.0``).
    max_basis_size : int
        Maximum basis size for diagonalisation (default ``10_000``).
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
    max_basis_size: int = 10_000
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_nqs_sqd(
    hamiltonian: Any,
    mol_info: Dict[str, Any],
    config: Optional[NQSSQDConfig] = None,
) -> SolverResult:
    """Execute the NQS+SQD pipeline.

    Stage 1 --- Train an autoregressive transformer NQS using variational
    Monte Carlo (energy minimisation via REINFORCE).

    Stage 2 --- Sample configurations from the trained NQS, deduplicate,
    and diagonalise the projected Hamiltonian.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian.
    mol_info : dict
        Molecular metadata.  Required keys: ``"n_orbitals"``,
        ``"n_alpha"``, ``"n_beta"``, ``"n_qubits"``.
    config : NQSSQDConfig or None
        Pipeline configuration.

    Returns
    -------
    SolverResult
        Energy, basis dimension, wall time, and metadata.
    """
    cfg = config or NQSSQDConfig()

    n_orb: int = mol_info["n_orbitals"]
    n_alpha: int = mol_info["n_alpha"]
    n_beta: int = mol_info["n_beta"]
    n_qubits: int = mol_info["n_qubits"]
    device = torch.device(cfg.device)

    logger.info(
        "run_nqs_sqd: %d orbitals, %d alpha, %d beta", n_orb, n_alpha, n_beta
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
    for epoch in range(cfg.nqs_train_epochs):
        optimiser.zero_grad()

        with torch.no_grad():
            configs = nqs.sample(256, temperature=cfg.temperature).to(device)

        alpha = configs[:, :n_orb]
        beta = configs[:, n_orb:]
        log_probs = nqs.log_prob(alpha, beta)

        # Local energies via diagonal elements
        local_energies = torch.zeros(configs.shape[0], device=device)
        for i in range(configs.shape[0]):
            local_energies[i] = hamiltonian.diagonal_element(configs[i])

        # REINFORCE loss
        baseline = local_energies.mean()
        loss = ((local_energies - baseline) * log_probs).mean()
        loss.backward()
        optimiser.step()

        train_energies.append(float(baseline.item()))

    nqs.eval()
    logger.info("  NQS training done, final E ~ %.6f", train_energies[-1])

    # --- Stage 2: Sample and diagonalise ---
    with torch.no_grad():
        sampled = nqs.sample(cfg.n_samples, temperature=cfg.temperature).to(device)

    basis = torch.unique(sampled, dim=0)
    if basis.shape[0] > cfg.max_basis_size:
        basis = basis[: cfg.max_basis_size]

    logger.info("  basis size: %d unique configs", basis.shape[0])

    energy, eigvec, occs = gpu_solve_fermion(basis, hamiltonian)

    wall_time = time.perf_counter() - t_start

    return SolverResult(
        energy=float(energy),
        diag_dim=int(basis.shape[0]),
        wall_time=wall_time,
        method="NQS+SQD",
        converged=True,
        metadata={
            "train_energies": train_energies,
            "n_unique_configs": int(basis.shape[0]),
        },
    )
