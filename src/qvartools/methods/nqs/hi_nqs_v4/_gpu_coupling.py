"""GPU-accelerated coupling <x|H|Psi_0> computation.

Drop-in replacement for _compute_coupling_to_ground_state in hi_nqs_sqd.py,
replacing the CPU dict lookup with torch.searchsorted.

Use-case: per-iter PT2 scoring, final PT2 correction. These calls dominate
the non-sqd time (~27% for default_100k).
"""
from __future__ import annotations

import numpy as np
import torch


def compute_coupling_gpu(new_candidates, sci_state, hamiltonian, n_orb, n_qubits):
    """Compute |<x|H|Psi_0>| for each candidate. GPU-native.

    new_candidates: list of (ibm_row, hash, config_tensor). Only config_tensor used.
    sci_state: has .amplitudes (n_a, n_b), .ci_strs_a, .ci_strs_b.

    Replaces the CPU dict lookup with torch.searchsorted on sorted keys.
    """
    if not new_candidates:
        return np.zeros(0)

    device = hamiltonian.device

    # Build sorted (key, amp) on GPU
    amps = np.asarray(sci_state.amplitudes)
    ci_strs_a = np.asarray(sci_state.ci_strs_a, dtype=np.int64)
    ci_strs_b = np.asarray(sci_state.ci_strs_b, dtype=np.int64)

    nonzero = np.argwhere(np.abs(amps) > 1e-14)
    if len(nonzero) == 0:
        return np.zeros(len(new_candidates))

    nz_ia = nonzero[:, 0]
    nz_ib = nonzero[:, 1]
    nz_amps = amps[nz_ia, nz_ib]
    nz_a_strs = ci_strs_a[nz_ia]
    nz_b_strs = ci_strs_b[nz_ib]

    MAX32 = 1 << 32
    nz_keys_cpu = (nz_a_strs.astype(np.int64) * MAX32 + nz_b_strs.astype(np.int64))
    order = np.argsort(nz_keys_cpu)
    sorted_keys = torch.from_numpy(nz_keys_cpu[order]).to(device)
    sorted_amps = torch.from_numpy(nz_amps[order]).double().to(device)

    # Batch all candidate configs
    all_configs = torch.stack([c[2] for c in new_candidates]).to(device)
    try:
        connected, elements, batch_idx = hamiltonian.get_connections_vectorized_batch(all_configs)
    except MemoryError:
        # Chunk recursively
        mid = len(new_candidates) // 2
        c1 = compute_coupling_gpu(new_candidates[:mid], sci_state, hamiltonian, n_orb, n_qubits)
        c2 = compute_coupling_gpu(new_candidates[mid:], sci_state, hamiltonian, n_orb, n_qubits)
        return np.concatenate([c1, c2])

    if len(connected) == 0:
        return np.zeros(len(new_candidates))

    # Encode connections → keys
    powers = (1 << torch.arange(n_orb, device=device, dtype=torch.long))
    conn_a = (connected[:, :n_orb].long() * powers).sum(dim=1)
    conn_b = (connected[:, n_orb:].long() * powers).sum(dim=1)
    conn_keys = conn_a * MAX32 + conn_b

    # GPU searchsorted lookup
    pos = torch.searchsorted(sorted_keys, conn_keys)
    pos_clamped = torch.clamp(pos, max=len(sorted_keys) - 1)
    valid = sorted_keys[pos_clamped] == conn_keys
    coeffs = torch.zeros_like(elements, dtype=torch.double)
    coeffs[valid] = sorted_amps[pos_clamped[valid]]

    weighted = elements.double() * coeffs
    coupling = torch.zeros(len(new_candidates), dtype=torch.double, device=device)
    coupling.scatter_add_(0, batch_idx.long(), weighted)

    return coupling.abs().cpu().numpy()
