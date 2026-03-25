"""
residual_config --- Configuration and helpers for residual-based expansion
=========================================================================

Provides the :class:`ResidualExpansionConfig` dataclass and shared helper
functions used by :class:`ResidualBasedExpander` and
:class:`SelectedCIExpander`.

Classes
-------
ResidualExpansionConfig
    Dataclass holding all hyperparameters for residual/CI expansion.

Functions
---------
_diagonalise_in_basis
    Solve the eigenvalue problem in a given configuration basis.
_generate_candidate_configs
    Generate candidate configurations connected to a current basis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from scipy.linalg import eigh as scipy_eigh

from qvartools.hamiltonians.hamiltonian import Hamiltonian

__all__ = [
    "ResidualExpansionConfig",
    "_diagonalise_in_basis",
    "_generate_candidate_configs",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualExpansionConfig:
    """Hyperparameters for residual-based and selected-CI basis expansion.

    Parameters
    ----------
    max_configs_per_iter : int
        Maximum number of new configurations to add per iteration.
    residual_threshold : float
        Minimum absolute residual component for a configuration to be
        considered for inclusion.
    max_iterations : int
        Maximum number of expansion iterations.
    max_basis_size : int
        Hard upper limit on the total basis size.
    min_energy_improvement_mha : float
        Minimum energy improvement in milliHartree per iteration.
        If the improvement drops below this for ``stagnation_patience``
        consecutive iterations, expansion terminates.
    stagnation_patience : int
        Number of consecutive iterations with insufficient energy
        improvement before early stopping.

    Examples
    --------
    >>> cfg = ResidualExpansionConfig(max_configs_per_iter=50)
    >>> cfg.max_configs_per_iter
    50
    """

    max_configs_per_iter: int = 100
    residual_threshold: float = 1e-4
    max_iterations: int = 10
    max_basis_size: int = 5000
    min_energy_improvement_mha: float = 0.05
    stagnation_patience: int = 3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_configs_per_iter < 1:
            raise ValueError(
                f"max_configs_per_iter must be >= 1, got {self.max_configs_per_iter}"
            )
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}"
            )
        if self.max_basis_size < 1:
            raise ValueError(
                f"max_basis_size must be >= 1, got {self.max_basis_size}"
            )
        if self.stagnation_patience < 1:
            raise ValueError(
                f"stagnation_patience must be >= 1, got {self.stagnation_patience}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diagonalise_in_basis(
    hamiltonian: Hamiltonian,
    basis: torch.Tensor,
    regularization: float = 1e-8,
    use_gpu: bool = True,
) -> Tuple[float, np.ndarray]:
    """Solve the eigenvalue problem in the given basis.

    When a CUDA device is available and ``use_gpu`` is ``True``, the
    Hamiltonian matrix is constructed and diagonalised on the GPU using
    :func:`torch.linalg.eigh`.  Otherwise, falls back to
    :func:`scipy.linalg.eigh` on the CPU.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The system Hamiltonian.
    basis : torch.Tensor
        Basis configurations, shape ``(n_basis, num_sites)``.
    regularization : float
        Tikhonov regularization for the overlap matrix.
    use_gpu : bool, optional
        If ``True`` (default), use GPU-accelerated ``torch.linalg.eigh``
        when CUDA is available.  If ``False``, always use scipy on CPU.

    Returns
    -------
    energy : float
        Lowest eigenvalue.
    eigenvector : np.ndarray
        Corresponding eigenvector, shape ``(n_basis,)``.
    """
    n = basis.shape[0]
    _use_cuda = use_gpu and torch.cuda.is_available()

    if _use_cuda:
        device = torch.device("cuda")
        basis_dev = basis.to(device)

        h_matrix = hamiltonian.matrix_elements(basis_dev, basis_dev).detach()
        h_matrix = h_matrix.to(dtype=torch.float64)

        # Symmetrise to guard against floating-point asymmetry
        h_sym = 0.5 * (h_matrix + h_matrix.T)

        # S = (1 + reg) * I, so L = sqrt(1+reg) * I and L^{-1} = 1/sqrt(1+reg) * I.
        # The standard eigenproblem reduces to H / (1+reg), which preserves
        # eigenvectors and scales eigenvalues.
        scale = 1.0 / (1.0 + regularization)
        h_standard = h_sym * scale
        h_standard = 0.5 * (h_standard + h_standard.T)

        eigenvalues, eigenvectors = torch.linalg.eigh(h_standard)

        return float(eigenvalues[0].item()), eigenvectors[:, 0].cpu().numpy()
    else:
        h_matrix = (
            hamiltonian.matrix_elements(basis, basis)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        s_matrix = (1.0 + regularization) * np.eye(n, dtype=np.float64)

        h_sym = 0.5 * (h_matrix + h_matrix.T)

        eigenvalues, eigenvectors = scipy_eigh(h_sym, s_matrix)
        return float(eigenvalues[0]), eigenvectors[:, 0]


def _generate_candidate_configs(
    hamiltonian: Hamiltonian,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Generate candidate configurations connected to the current basis.

    For each configuration in the basis, find all states connected via the
    Hamiltonian and collect those not already in the basis.

    Uses integer hash-based set membership for O(1) lookups instead of
    tuple conversion.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The system Hamiltonian.
    basis : torch.Tensor
        Current basis configurations, shape ``(n_basis, num_sites)``.

    Returns
    -------
    torch.Tensor
        Candidate configurations not in the current basis,
        shape ``(n_candidates, num_sites)``.
    """
    candidates: List[torch.Tensor] = []

    # Pre-move basis to CPU once for Numba-based get_connections
    basis_cpu = basis.detach().cpu()

    for i in range(basis_cpu.shape[0]):
        connected, _ = hamiltonian.get_connections(basis_cpu[i])
        if connected.numel() > 0:
            candidates.append(connected)

    if not candidates:
        return torch.zeros(0, basis.shape[1], dtype=torch.int64)

    all_candidates = torch.cat(candidates, dim=0)
    all_candidates = torch.unique(all_candidates, dim=0)

    # Remove configs already in the basis using vectorised integer hashing
    # instead of slow Python tuple conversion
    n_sites = basis.shape[1]
    powers = torch.tensor(
        [1 << k for k in range(n_sites - 1, -1, -1)],
        dtype=torch.int64,
    )

    basis_hashes = (basis_cpu.to(torch.int64) * powers.unsqueeze(0)).sum(dim=-1)
    basis_hash_set = set(basis_hashes.tolist())

    cand_hashes = (
        all_candidates.to(torch.int64) * powers.unsqueeze(0)
    ).sum(dim=-1)
    mask = torch.tensor(
        [int(h.item()) not in basis_hash_set for h in cand_hashes],
        dtype=torch.bool,
    )

    if mask.any():
        return all_candidates[mask]
    return torch.zeros(0, basis.shape[1], dtype=torch.int64)
