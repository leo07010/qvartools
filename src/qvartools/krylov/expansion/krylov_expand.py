"""
krylov_expand --- Basis expansion via Hamiltonian connections
============================================================

Provides :func:`expand_basis_via_connections`, which grows a configuration
basis by following the off-diagonal structure of the Hamiltonian.  Starting
from a set of reference configurations, the function collects all
Hamiltonian-connected states, ranks them by coupling strength, and adds
unique new configurations up to a specified cap.

Supports two-hop expansion: first hop discovers singles/doubles from seed
configs, second hop expands from those to reach up to quadruples.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

__all__ = [
    "expand_basis_via_connections",
]

logger = logging.getLogger(__name__)


def _select_reference_configs(
    basis: torch.Tensor, hamiltonian: Any, n_ref: int
) -> torch.Tensor:
    """Select reference configurations with the lowest diagonal energy.

    Parameters
    ----------
    basis : torch.Tensor
        Current basis configurations, shape ``(n_basis, n_sites)``.
    hamiltonian : Hamiltonian
        The system Hamiltonian (must implement ``diagonal_element``).
    n_ref : int
        Number of reference configurations to select.

    Returns
    -------
    torch.Tensor
        Selected reference configurations, shape ``(n_ref, n_sites)``.
    """
    n_basis = basis.shape[0]
    n_ref = min(n_ref, n_basis)

    if n_ref == n_basis:
        return basis

    diag_energies = torch.zeros(n_basis, dtype=torch.float64)
    for i in range(n_basis):
        diag_energies[i] = hamiltonian.diagonal_element(basis[i])

    _, indices = torch.topk(diag_energies, n_ref, largest=False)
    return basis[indices]


def expand_basis_via_connections(
    basis: torch.Tensor,
    hamiltonian: Any,
    max_new: int = 500,
    n_ref: "int | None" = None,
    coupling_rank: bool = True,
) -> torch.Tensor:
    """Expand a configuration basis by following Hamiltonian connections.

    Selects ``n_ref`` reference configurations from the basis, collects
    all states connected to them via the Hamiltonian, ranks by coupling
    strength, removes duplicates and states already in the basis, and
    returns the expanded basis with up to ``max_new`` new configurations.

    Performs a two-hop expansion when budget allows: first hop discovers
    single/double excitations from seed configs, second hop expands from
    those to reach quadruples.

    Parameters
    ----------
    basis : torch.Tensor
        Current basis configurations, shape ``(n_basis, n_sites)`` with
        integer entries.
    hamiltonian
        The system Hamiltonian.  Must implement ``diagonal_element`` and
        ``get_connections``.
    max_new : int, optional
        Maximum number of new configurations to add (default ``500``).
    n_ref : int or None, optional
        Number of reference configurations. Defaults to
        ``min(len(basis), 50)``.
    coupling_rank : bool, optional
        If ``True``, rank new configs by max ``|H_ij|`` coupling
        strength and keep top ``max_new`` (default ``True``).

    Returns
    -------
    torch.Tensor
        Expanded basis configurations, shape
        ``(n_basis + n_added, n_sites)`` where ``n_added <= max_new``.
    """
    if basis.shape[0] == 0:
        logger.warning("expand_basis_via_connections: empty basis.")
        return basis

    if max_new < 1:
        return basis

    if isinstance(basis, np.ndarray):
        basis = torch.from_numpy(basis).long()
    basis = basis.cpu().long()

    existing_keys = {row.tobytes() for row in basis.numpy()}

    if n_ref is None:
        n_ref = min(len(basis), 50)

    refs = _select_reference_configs(basis, hamiltonian, n_ref)

    # First hop
    new_map, new_configs_map = _collect_connections(
        refs, hamiltonian, existing_keys
    )

    if not new_map:
        logger.debug(
            "expand_basis_via_connections: no new configs from %d refs.",
            refs.shape[0],
        )
        return basis

    keys_list = list(new_map.keys())
    new_configs_list = [new_configs_map[k] for k in keys_list]
    new_couplings = np.array([new_map[k] for k in keys_list])
    new_tensor = torch.stack(new_configs_list)

    new_tensor, new_couplings = _truncate_by_coupling(
        new_tensor, new_couplings, max_new, coupling_rank
    )

    expanded = torch.cat([basis, new_tensor], dim=0)

    # Second hop (if budget allows)
    if len(new_tensor) > 0 and max_new > len(new_tensor):
        remaining = max_new - len(new_tensor)
        second_refs = new_tensor[: min(50, len(new_tensor))]
        hop2_map, hop2_configs_map = _collect_connections(
            second_refs, hamiltonian, existing_keys
        )
        if hop2_map:
            keys2 = list(hop2_map.keys())
            hop2_tensor = torch.stack([hop2_configs_map[k] for k in keys2])
            hop2_couplings = np.array([hop2_map[k] for k in keys2])
            hop2_tensor, _ = _truncate_by_coupling(
                hop2_tensor, hop2_couplings, remaining, coupling_rank
            )
            if len(hop2_tensor) > 0:
                expanded = torch.cat([expanded, hop2_tensor], dim=0)

    n_added = expanded.shape[0] - basis.shape[0]
    logger.info(
        "expand_basis_via_connections: added %d configs "
        "(basis %d -> %d, %d refs).",
        n_added,
        basis.shape[0],
        expanded.shape[0],
        refs.shape[0],
    )

    return expanded


def _collect_connections(
    refs: torch.Tensor,
    hamiltonian: Any,
    existing_keys: set,
) -> "tuple[dict, dict]":
    """Collect connected configurations from reference states.

    For each reference configuration, retrieves Hamiltonian-connected
    states via ``hamiltonian.get_connections`` and tracks the maximum
    coupling strength ``|H_ij|`` for each new configuration.

    Parameters
    ----------
    refs : torch.Tensor
        Reference configurations, shape ``(n_ref, n_sites)``.
    hamiltonian
        The system Hamiltonian (must implement ``get_connections``).
    existing_keys : set
        Set of byte-keys for configurations already in the basis.
        **Modified in-place**: newly discovered keys are added.

    Returns
    -------
    new_map : dict
        Mapping ``config_key (bytes) -> max |H_ij|`` coupling strength.
    new_configs_map : dict
        Mapping ``config_key (bytes) -> config tensor``.
    """
    new_map: dict[bytes, float] = {}
    new_configs_map: dict[bytes, torch.Tensor] = {}

    for ref in refs:
        try:
            connected, elements = hamiltonian.get_connections(ref)
        except Exception as e:
            logger.debug("get_connections failed: %s", e)
            continue

        if connected is None or len(connected) == 0:
            continue

        connected = connected.cpu().long()
        if elements is not None:
            elements_np = elements.detach().cpu().numpy()
        else:
            elements_np = np.ones(len(connected))

        for i in range(len(connected)):
            key = connected[i].numpy().tobytes()
            if key in existing_keys:
                continue
            coupling = abs(float(elements_np[i]))

            if key not in new_map or coupling > new_map[key]:
                new_map[key] = coupling
                new_configs_map[key] = connected[i]

    existing_keys.update(new_map.keys())
    return new_map, new_configs_map


def _truncate_by_coupling(
    tensor: torch.Tensor,
    couplings: np.ndarray,
    max_new: int,
    coupling_rank: bool,
) -> "tuple[torch.Tensor, np.ndarray]":
    """Keep at most ``max_new`` configurations ranked by coupling strength.

    Parameters
    ----------
    tensor : torch.Tensor
        Candidate configurations, shape ``(n_candidates, n_sites)``.
    couplings : np.ndarray
        Coupling strengths, shape ``(n_candidates,)``.
    max_new : int
        Maximum number of configurations to retain.
    coupling_rank : bool
        If ``True``, sort by descending coupling and keep the top
        ``max_new``.  If ``False``, keep the first ``max_new`` in
        their original order.

    Returns
    -------
    tensor : torch.Tensor
        Retained configurations, shape ``(<= max_new, n_sites)``.
    couplings : np.ndarray
        Corresponding coupling strengths.
    """
    if len(tensor) <= max_new:
        return tensor, couplings
    if coupling_rank:
        top_idx = np.argsort(couplings)[::-1][:max_new].copy()
    else:
        top_idx = np.arange(max_new)
    return tensor[top_idx], couplings[top_idx]
