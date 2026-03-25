"""
Spectral range utilities for SKQD time step computation.

Provides shared function to compute the optimal Krylov time step
from the spectral range of a molecular Hamiltonian's subspace.

Per SKQD paper (Theorem 3.1, Epperly et al.):
    dt_optimal = pi / (E_max - E_min)

Four strategies for different config-space sizes:
  - Small (<=5K): Dense diagonalization
  - Medium (5K-20K): Sparse eigsh on full matrix (built once via matrix_elements_fast)
  - Medium-large (20K-200K): Sparse batched eigsh (GPU-vectorized COO build)
  - Very large (>200K): Diagonal approximation O(n_configs)
"""

from __future__ import annotations

import logging
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from qvartools.hamiltonians.molecular.hamiltonian import MolecularHamiltonian

__all__ = [
    "compute_optimal_dt",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basis enumeration
# ---------------------------------------------------------------------------

def _enumerate_basis(hamiltonian: MolecularHamiltonian) -> torch.Tensor:
    """Enumerate all particle-conserving configurations."""
    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    device = hamiltonian.device

    alpha_configs = list(combinations(range(n_orb), n_alpha))
    beta_configs = list(combinations(range(n_orb), n_beta))

    basis = []
    for ac in alpha_configs:
        for bc in beta_configs:
            cfg = torch.zeros(2 * n_orb, dtype=torch.long, device=device)
            for o in ac:
                cfg[o] = 1
            for o in bc:
                cfg[o + n_orb] = 1
            basis.append(cfg)
    return torch.stack(basis)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _spectral_range_dense(
    hamiltonian: MolecularHamiltonian,
    basis_tensor: torch.Tensor,
) -> float:
    """Dense diagonalization for small subspaces (<=5K configs)."""
    h_sub = hamiltonian.matrix_elements(basis_tensor, basis_tensor)
    h_np = h_sub.cpu().numpy().real.astype(np.float64)
    h_np = 0.5 * (h_np + h_np.T)
    evals = np.linalg.eigvalsh(h_np)
    return float(evals[-1] - evals[0])


def _spectral_range_sparse(
    hamiltonian: MolecularHamiltonian,
    basis_tensor: torch.Tensor,
) -> float:
    """Sparse eigsh on full matrix for medium subspaces (5K-20K).

    Builds the full matrix once via matrix_elements_fast (optimized Hermitian
    construction), converts to sparse, then runs eigsh for extremal eigenvalues.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh as scipy_eigsh

    # Use matrix_elements_fast (bra==ket optimized path) instead of
    # matrix_elements(bra, ket) which takes the slow general path
    h_sub = hamiltonian.matrix_elements_fast(basis_tensor)
    h_gpu = h_sub.to(dtype=torch.float64)
    h_gpu = 0.5 * (h_gpu + h_gpu.T)

    h_np = h_gpu.cpu().numpy()
    h_sp = csr_matrix(h_np)
    del h_np, h_gpu

    e_min = float(scipy_eigsh(h_sp, k=1, which="SA", return_eigenvectors=False)[0])
    e_max = float(scipy_eigsh(h_sp, k=1, which="LA", return_eigenvectors=False)[0])
    return e_max - e_min


def _spectral_range_sparse_batched(
    hamiltonian: MolecularHamiltonian,
    basis_tensor: torch.Tensor,
) -> float:
    """
    Sparse eigsh for medium-large config spaces (20K-200K).

    Builds a sparse COO matrix in batches using GPU-vectorized
    get_connections_vectorized_batch (same engine as matrix_elements_fast),
    then runs eigsh for extremal eigenvalues. Avoids the O(n**2) dense matrix.

    For 108K configs with ~100 connections each -> ~10M nonzeros -> ~240 MB sparse.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh as scipy_eigsh

    n = len(basis_tensor)
    device = basis_tensor.device

    # Integer-encode all basis configs for O(1) lookup via searchsorted
    n_sites = basis_tensor.shape[1]
    powers = torch.tensor(
        [1 << k for k in range(n_sites - 1, -1, -1)],
        dtype=torch.int64,
        device=device,
    )
    config_ints = (basis_tensor.long() * powers.unsqueeze(0)).sum(dim=1)
    sorted_config_ints, sort_order = config_ints.sort()

    # Collect COO entries: rows, cols, vals
    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []

    # Diagonal elements (vectorized)
    diag = hamiltonian.diagonal_elements_batch(basis_tensor).to(torch.float64)
    diag_idx = np.arange(n)
    all_rows.append(diag_idx)
    all_cols.append(diag_idx)
    all_vals.append(diag.cpu().numpy())

    # Off-diagonal: process in batches to limit GPU memory
    batch_size = min(5000, n)
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        batch = basis_tensor[i_start:i_end]

        connected, elements, batch_indices = hamiltonian.get_connections_vectorized_batch(batch)
        if len(connected) == 0:
            continue

        # Map connected configs to basis indices via searchsorted
        conn_ints = (connected.long() * powers.unsqueeze(0)).sum(dim=1)
        search_pos = torch.searchsorted(sorted_config_ints, conn_ints)
        search_pos_clamped = search_pos.clamp(max=n - 1)
        match_mask = sorted_config_ints[search_pos_clamped] == conn_ints

        if not match_mask.any():
            continue

        valid_k = match_mask.nonzero(as_tuple=True)[0]
        # row = matched basis index, col = source config (offset by batch start)
        rows = sort_order[search_pos_clamped[valid_k]].cpu().numpy()
        cols = (batch_indices[valid_k] + i_start).cpu().numpy()
        vals = elements[valid_k].to(torch.float64).cpu().numpy()

        # Filter out diagonal entries and negligible values
        off_diag = rows != cols
        significant = np.abs(vals) > 1e-14
        mask = off_diag & significant
        all_rows.append(rows[mask])
        all_cols.append(cols[mask])
        all_vals.append(vals[mask])

    # Build sparse matrix
    rows_cat = np.concatenate(all_rows)
    cols_cat = np.concatenate(all_cols)
    vals_cat = np.concatenate(all_vals)

    h_sp = coo_matrix((vals_cat, (rows_cat, cols_cat)), shape=(n, n)).tocsr()
    # Enforce Hermitian symmetry
    h_sp = 0.5 * (h_sp + h_sp.T)

    del rows_cat, cols_cat, vals_cat, basis_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(
        "Sparse matrix: %s nonzeros (%d MB)",
        f"{h_sp.nnz:,}",
        h_sp.nnz * 8 // (1024 ** 2),
    )

    try:
        e_min = float(scipy_eigsh(h_sp, k=1, which="SA", return_eigenvectors=False)[0])
        e_max = float(scipy_eigsh(h_sp, k=1, which="LA", return_eigenvectors=False)[0])
        return e_max - e_min
    except Exception:
        logger.warning(
            "Sparse batched eigsh failed, falling back to diagonal approximation",
            exc_info=True,
        )
        return _spectral_range_diagonal(hamiltonian, _enumerate_basis(hamiltonian))


def _spectral_range_diagonal(
    hamiltonian: MolecularHamiltonian,
    basis_tensor: torch.Tensor,
) -> float:
    """
    Diagonal approximation for large config spaces (>20K).

    Estimates spectral range from diagonal elements H_ii only.
    O(n_configs) time and memory. Less accurate but always feasible.
    """
    n = len(basis_tensor)
    batch_size = 5000

    diag_min = float("inf")
    diag_max = float("-inf")

    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        batch = basis_tensor[i_start:i_end]
        # Use diagonal_elements_batch directly -- O(batch) instead of O(batch**2)
        diag_vals = hamiltonian.diagonal_elements_batch(batch).cpu().numpy()
        diag_min = min(diag_min, float(diag_vals.min()))
        diag_max = max(diag_max, float(diag_vals.max()))

    spectral_range = diag_max - diag_min

    # Diagonal approximation tends to underestimate; apply safety factor
    return spectral_range * 1.2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_optimal_dt(hamiltonian: MolecularHamiltonian) -> Tuple[float, float]:
    """
    Compute optimal Krylov time step from spectral range of subspace Hamiltonian.

    Enumerates all particle-conserving configurations, builds the subspace
    Hamiltonian, and computes its spectral range to derive the optimal dt.

    Per SKQD paper (Theorem 3.1, Epperly et al.):
        dt_optimal = pi / (E_max - E_min)

    Strategy selection by config-space size:
      <=5K:      Dense eigvalsh (exact, fast)
      5K-20K:    Sparse eigsh on full matrix (built once via matrix_elements_fast)
      20K-200K:  Sparse batched eigsh (GPU-vectorized COO build, no dense matrix)
      >200K:     Diagonal approximation with 1.2x safety factor

    Args:
        hamiltonian: MolecularHamiltonian with n_orbitals, n_alpha, n_beta attributes

    Returns:
        (optimal_dt, spectral_range) where optimal_dt = pi / spectral_range
    """
    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    n = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

    if n > 200_000:
        # Very large: diagonal approximation -- O(n) time
        logger.info("Spectral range: diagonal approximation (%s configs)", f"{n:,}")
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_diagonal(hamiltonian, basis_tensor)
    elif n > 20_000:
        # Medium-large (20K-200K): build sparse matrix in batches, then eigsh
        logger.info("Spectral range: sparse batched eigsh (%s configs)", f"{n:,}")
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_sparse_batched(hamiltonian, basis_tensor)
    elif n > 5000:
        # Medium: build full matrix once via optimized matrix_elements_fast,
        # then sparse eigsh.
        logger.info("Spectral range: sparse eigsh (%s configs)", f"{n:,}")
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_sparse(hamiltonian, basis_tensor)
    else:
        basis_tensor = _enumerate_basis(hamiltonian)
        spectral_range = _spectral_range_dense(hamiltonian, basis_tensor)

    optimal_dt = np.pi / spectral_range
    return optimal_dt, spectral_range
