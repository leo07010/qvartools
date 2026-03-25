"""
projected_hamiltonian --- Build projected Hamiltonian in a sampled basis
========================================================================

Constructs the projected Hamiltonian matrix H_ij = <x_i|H|x_j> in a
sampled computational-basis subset.  Uses the Hamiltonian's
:meth:`~qvartools.hamiltonians.hamiltonian.Hamiltonian.get_connections` interface
for sparse construction and hash-based lookup for O(1) basis membership
testing.

Classes
-------
ProjectedHamiltonianConfig
    Hyperparameters controlling the projected construction.
ProjectedHamiltonianBuilder
    Builds the sparse projected Hamiltonian from a basis set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import scipy.sparse
import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian

__all__ = [
    "ProjectedHamiltonianConfig",
    "ProjectedHamiltonianBuilder",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectedHamiltonianConfig:
    """Hyperparameters for projected Hamiltonian construction.

    Parameters
    ----------
    use_sparse : bool
        If ``True`` (default), build a sparse CSR matrix.  If ``False``,
        build a dense NumPy array (only practical for small bases).
    batch_size : int
        Number of basis states to process per batch when constructing
        the matrix.  Larger batches use more memory but may be faster.

    Examples
    --------
    >>> cfg = ProjectedHamiltonianConfig(use_sparse=False)
    >>> cfg.use_sparse
    False
    """

    use_sparse: bool = True
    batch_size: int = 1000

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


# ---------------------------------------------------------------------------
# Hash-based index lookup
# ---------------------------------------------------------------------------


def _build_config_index(basis_states: torch.Tensor) -> Dict[int, int]:
    """Build a hash map from configuration hash to basis index.

    Each configuration is converted to a unique integer hash by treating
    the binary vector as digits in a base equal to the maximum value + 1.

    Parameters
    ----------
    basis_states : torch.Tensor
        Basis configurations, shape ``(n_basis, n_sites)``.

    Returns
    -------
    dict
        Mapping from configuration hash to row/column index.
    """
    n_basis, n_sites = basis_states.shape
    # Use a large prime-based hash to avoid collisions
    # Treat each config as a number in base (max_val+1)
    max_val = int(basis_states.max().item()) + 1
    multipliers = torch.tensor(
        [max_val ** i for i in range(n_sites - 1, -1, -1)],
        dtype=torch.int64,
        device=basis_states.device,
    )
    hashes = (basis_states.to(torch.int64) * multipliers.unsqueeze(0)).sum(dim=1)
    return {int(hashes[i].item()): i for i in range(n_basis)}


def _config_hash(config: torch.Tensor, max_val: int) -> int:
    """Compute the hash of a single configuration.

    Parameters
    ----------
    config : torch.Tensor
        Configuration vector, shape ``(n_sites,)``.
    max_val : int
        Base for the positional encoding (typically ``max(config) + 1``
        or ``2`` for binary).

    Returns
    -------
    int
        Integer hash of the configuration.
    """
    n_sites = config.shape[0]
    h = 0
    for i in range(n_sites):
        h = h * max_val + int(config[i].item())
    return h


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ProjectedHamiltonianBuilder:
    """Build the projected Hamiltonian H_ij = <x_i|H|x_j> in a sampled basis.

    Uses the Hamiltonian's ``get_connections()`` method for efficient sparse
    construction: for each basis state |x_j>, the connected states and
    their matrix elements are retrieved, and a hash-based lookup determines
    which connections lie within the sampled basis.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian operator (must implement ``diagonal_element`` and
        ``get_connections``).
    config : ProjectedHamiltonianConfig or None, optional
        Construction hyperparameters.  If ``None``, defaults are used.

    Examples
    --------
    >>> from qvartools.hamiltonians.spin import HeisenbergHamiltonian
    >>> ham = HeisenbergHamiltonian(num_sites=6, J=1.0)
    >>> builder = ProjectedHamiltonianBuilder(ham)
    >>> basis = torch.tensor([[1,0,1,0,1,0], [0,1,0,1,0,1]])
    >>> H_proj = builder.build(basis)
    >>> H_proj.shape
    (2, 2)
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        config: Optional[ProjectedHamiltonianConfig] = None,
    ) -> None:
        self._hamiltonian = hamiltonian
        self._config = config if config is not None else ProjectedHamiltonianConfig()

    def build(
        self,
        basis_states: torch.Tensor,
    ) -> scipy.sparse.csr_matrix:
        """Construct the projected Hamiltonian matrix.

        Uses vectorised batch diagonal computation and hash-based
        connection matching for efficiency.

        Parameters
        ----------
        basis_states : torch.Tensor
            Sampled basis configurations, shape ``(n_basis, n_sites)``.
            Each row is a computational-basis state.

        Returns
        -------
        scipy.sparse.csr_matrix
            Projected Hamiltonian of shape ``(n_basis, n_basis)``.

        Raises
        ------
        ValueError
            If ``basis_states`` has fewer than 1 row or the wrong number
            of columns.
        """
        n_basis, n_sites = basis_states.shape

        if n_basis < 1:
            raise ValueError("basis_states must contain at least one state.")
        if n_sites != self._hamiltonian.num_sites:
            raise ValueError(
                f"basis_states has {n_sites} sites, but Hamiltonian expects "
                f"{self._hamiltonian.num_sites}."
            )

        logger.info(
            "Building projected Hamiltonian: %d basis states, %d sites",
            n_basis,
            n_sites,
        )

        # Build hash-based index for O(1) membership lookup
        max_val = max(int(basis_states.max().item()) + 1, 2)
        config_index = _build_config_index(basis_states)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        # --- Vectorised diagonal computation (batch, no Python loop) ---
        if hasattr(self._hamiltonian, "diagonal_elements_batch"):
            diag_all = self._hamiltonian.diagonal_elements_batch(basis_states)
            for j in range(n_basis):
                diag_float = float(diag_all[j].item())
                if abs(diag_float) > 0:
                    rows.append(j)
                    cols.append(j)
                    vals.append(diag_float)
        else:
            # Fallback for Hamiltonians without batch diagonal
            for j in range(n_basis):
                diag_val = self._hamiltonian.diagonal_element(basis_states[j])
                diag_float = float(
                    diag_val.real if hasattr(diag_val, "real") else diag_val
                )
                if abs(diag_float) > 0:
                    rows.append(j)
                    cols.append(j)
                    vals.append(diag_float)

        # --- Off-diagonal: connected states ---
        # Pre-move basis to CPU once for Numba-based get_connections
        basis_cpu = basis_states.detach().cpu()

        # Pre-sort hashes for vectorised searchsorted matching
        hash_values = torch.tensor(
            list(config_index.keys()), dtype=torch.int64
        )
        hash_indices = torch.tensor(
            list(config_index.values()), dtype=torch.int64
        )
        sorted_order = torch.argsort(hash_values)
        sorted_hash_keys = hash_values[sorted_order]
        sorted_hash_vals = hash_indices[sorted_order]
        n_hash = sorted_hash_keys.shape[0]

        for j in range(n_basis):
            connected, elements = self._hamiltonian.get_connections(basis_cpu[j])

            if connected.numel() == 0:
                continue

            # Vectorised hash matching via searchsorted
            if hasattr(self._hamiltonian, "_config_hash_batch"):
                conn_hashes = self._hamiltonian._config_hash_batch(connected)
            else:
                conn_hashes = torch.tensor(
                    [_config_hash(connected[c], max_val) for c in range(connected.shape[0])],
                    dtype=torch.int64,
                )

            positions = torch.searchsorted(sorted_hash_keys, conn_hashes)
            positions = positions.clamp(max=n_hash - 1)
            matched_mask = sorted_hash_keys[positions] == conn_hashes

            if matched_mask.any():
                matched_i = sorted_hash_vals[positions[matched_mask]]
                matched_els = elements[matched_mask]
                for k in range(matched_i.shape[0]):
                    mel = float(matched_els[k].item())
                    if abs(mel) > 0:
                        rows.append(int(matched_i[k].item()))
                        cols.append(j)
                        vals.append(mel)

            if j > 0 and j % 1000 == 0:
                logger.debug(
                    "Processed %d / %d states (%d non-zeros so far)",
                    j,
                    n_basis,
                    len(vals),
                )

        H_proj = scipy.sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_basis, n_basis),
        )

        logger.info(
            "Projected Hamiltonian: shape (%d, %d), nnz=%d, density=%.4f",
            n_basis,
            n_basis,
            H_proj.nnz,
            H_proj.nnz / (n_basis * n_basis) if n_basis > 0 else 0,
        )

        return H_proj
