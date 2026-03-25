"""
bitstring_format --- Format conversion between qvartools and IBM bitstring formats
===================================================================================

Provides vectorized conversion between the qvartools configuration format
``[alpha_0, ..., alpha_{n-1}, beta_0, ..., beta_{n-1}]`` and the IBM
bitstring format used by ``qiskit-addon-sqd``.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "hash_config",
    "configs_to_ibm_format",
    "ibm_format_to_configs",
    "vectorized_dedup",
]


def hash_config(config: torch.Tensor) -> int:
    """Hash a binary configuration to a unique integer.

    Treats the configuration as a big-endian bitstring and computes
    its integer value.

    Parameters
    ----------
    config : torch.Tensor
        Binary vector, shape ``(n_sites,)`` with entries in ``{0, 1}``.

    Returns
    -------
    int
        Integer hash of the configuration.
    """
    n = config.shape[0]
    val = 0
    for i in range(n):
        val = val * 2 + int(config[i].item())
    return val


def configs_to_ibm_format(
    configs: torch.Tensor | np.ndarray,
    n_orb: int,
    n_qubits: int,
) -> np.ndarray:
    """Convert config tensor to IBM bitstring matrix.

    Parameters
    ----------
    configs : torch.Tensor or np.ndarray
        ``(n_configs, 2*n_orb)`` array in qvartools format.
    n_orb : int
        Number of spatial orbitals.
    n_qubits : int
        Total qubit count (``2 * n_orb``).

    Returns
    -------
    np.ndarray
        ``(n_configs, n_qubits)`` bool array in IBM format.
    """
    if isinstance(configs, torch.Tensor):
        configs_np = configs.cpu().numpy()
    else:
        configs_np = np.asarray(configs)

    n = len(configs_np)
    if n == 0:
        return np.zeros((0, n_qubits), dtype=bool)

    bs = np.zeros((n, n_qubits), dtype=bool)
    bs[:, :n_orb] = configs_np[:, :n_orb][:, ::-1].astype(bool)
    bs[:, n_orb:] = configs_np[:, n_orb:][:, ::-1].astype(bool)
    return bs


def ibm_format_to_configs(
    bs_matrix: np.ndarray,
    n_orb: int,
    n_qubits: int,
) -> torch.Tensor:
    """Convert IBM bitstring matrix back to qvartools config tensor.

    Parameters
    ----------
    bs_matrix : np.ndarray
        ``(n_configs, n_qubits)`` bool array in IBM format.
    n_orb : int
        Number of spatial orbitals.
    n_qubits : int
        Total qubit count (``2 * n_orb``).

    Returns
    -------
    torch.Tensor
        ``(n_configs, n_qubits)`` long tensor in qvartools format.
    """
    bs = np.asarray(bs_matrix)
    n = len(bs)
    if n == 0:
        return torch.zeros(0, n_qubits, dtype=torch.long)

    configs = np.zeros((n, n_qubits), dtype=np.int64)
    configs[:, :n_orb] = bs[:, :n_orb][:, ::-1].astype(np.int64)
    configs[:, n_orb:] = bs[:, n_orb:][:, ::-1].astype(np.int64)
    return torch.from_numpy(configs)


def vectorized_dedup(
    existing_bs: np.ndarray | None,
    new_bs: np.ndarray,
) -> np.ndarray:
    """Return rows in *new_bs* not present in *existing_bs*.

    Parameters
    ----------
    existing_bs : np.ndarray or None
        ``(n_existing, n_cols)`` bool array, or ``None`` if empty.
    new_bs : np.ndarray
        ``(n_new, n_cols)`` bool array.

    Returns
    -------
    np.ndarray
        Truly-new rows (preserving order from *new_bs*).
    """
    if len(new_bs) == 0:
        return new_bs

    n_cols = new_bs.shape[1]

    _, first_idx = np.unique(
        np.ascontiguousarray(new_bs).view(
            np.dtype((np.void, new_bs.dtype.itemsize * n_cols))
        ),
        return_index=True,
    )
    first_idx.sort()
    new_unique = new_bs[first_idx]

    if existing_bs is None or len(existing_bs) == 0:
        return new_unique

    existing_set = {row.tobytes() for row in np.ascontiguousarray(existing_bs)}
    mask = np.array(
        [row.tobytes() not in existing_set for row in np.ascontiguousarray(new_unique)],
        dtype=bool,
    )
    return new_unique[mask]
