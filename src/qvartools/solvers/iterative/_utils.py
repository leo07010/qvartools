"""
_iterative_utils --- Shared helpers for iterative NF solvers
============================================================

Provides the default training configuration and shared helper functions
used by :class:`~qvartools.solvers.iterative_sqd.IterativeNFSQDSolver`
and :class:`~qvartools.solvers.iterative_skqd.IterativeNFSKQDSolver`.

Functions
---------
_create_flow
    Instantiate the normalizing flow for a given Hamiltonian.
_bias_nqs
    Bias the NQS toward the previous eigenvector distribution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian

__all__ = [
    "_DEFAULT_TRAINING_CONFIG",
    "_create_flow",
    "_bias_nqs",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------------

_DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "samples_per_batch": 300,
    "num_batches": 5,
    "num_epochs": 100,
    "min_epochs": 20,
    "convergence_threshold": 0.01,
    "flow_lr": 1e-3,
    "nqs_lr": 1e-3,
    "teacher_weight": 1.0,
    "physics_weight": 0.1,
    "entropy_weight": 0.01,
    "initial_temperature": 1.5,
    "final_temperature": 0.1,
    "temperature_decay_epochs": 60,
    "inject_essential_configs": True,
}


# ---------------------------------------------------------------------------
# Shared helper: create flow
# ---------------------------------------------------------------------------


def _create_flow(
    hamiltonian: Hamiltonian, n_qubits: int
) -> torch.nn.Module:
    """Instantiate the normalizing flow.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Used to extract particle numbers for conserving flows.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    torch.nn.Module
        Flow sampler instance.
    """
    if hasattr(hamiltonian, "integrals"):
        from qvartools.flows import ParticleConservingFlowSampler

        integrals = hamiltonian.integrals
        return ParticleConservingFlowSampler(
            num_sites=n_qubits,
            n_alpha=integrals.n_alpha,
            n_beta=integrals.n_beta,
        )

    from qvartools.flows import DiscreteFlowSampler

    return DiscreteFlowSampler(num_sites=n_qubits)


def _bias_nqs(
    nqs: torch.nn.Module,
    basis: torch.Tensor,
    eigenvector: np.ndarray,
) -> None:
    """Bias the NQS toward the previous eigenvector distribution.

    Performs a few gradient steps on the NQS to match the
    Born-rule distribution implied by the eigenvector coefficients.

    Parameters
    ----------
    nqs : torch.nn.Module
        The neural quantum state to bias.
    basis : torch.Tensor
        Basis configurations, shape ``(n_basis, n_qubits)``.
    eigenvector : np.ndarray
        Eigenvector coefficients, shape ``(n_basis,)``.
    """
    target_log_amp = torch.tensor(
        np.log(np.abs(eigenvector) + 1e-30),
        dtype=torch.float32,
        device=basis.device,
    )

    optimizer = torch.optim.Adam(nqs.parameters(), lr=1e-3)

    for _ in range(50):
        optimizer.zero_grad()
        log_amp = nqs.log_amplitude(basis.float())
        loss = torch.nn.functional.mse_loss(log_amp, target_log_amp)
        loss.backward()
        optimizer.step()
