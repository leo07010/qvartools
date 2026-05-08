"""Single-run multi-GPU PT2 parallelism.

Provides:
  - replicate_hamiltonian(base, devices) : clone a MolecularHamiltonian across K GPUs
  - compute_coupling_multi_gpu(...)       : per-iter PT2 scoring across K GPUs
                                            (splits candidate pool)
  - final_pt2_multi_gpu(...)              : final PT2 correction across K GPUs
                                            (splits external space)

Safe to call when K=1: falls back to single-GPU path.

Correctness note: for PT2, each external det's coupling is computed independently
over ALL basis dets (via get_connections from that external). Splitting externals
across GPUs is therefore exact — no cross-terms to lose. Splitting *source* dets
would be wrong because multiple sources contribute to the same external's coupling,
but that is NOT what this module does.
"""
from __future__ import annotations

import concurrent.futures
from typing import List, Optional

import numpy as np
import torch

from ..hamiltonians.molecular import MolecularHamiltonian
from ._gpu_coupling import compute_coupling_gpu
from ._core import _configs_to_ibm_format, _ibm_format_to_configs


# -----------------------------------------------------------------------------
def visible_gpus():
    n = torch.cuda.device_count()
    return [torch.device(f"cuda:{i}") for i in range(n)]


def replicate_hamiltonian(base_hamiltonian, devices: List[torch.device]):
    """Return list of hamiltonians, one per device. Re-uses base for its own device."""
    hams = []
    base_dev = torch.device(base_hamiltonian.device)
    for d in devices:
        if base_dev == d:
            hams.append(base_hamiltonian)
        else:
            hams.append(MolecularHamiltonian(base_hamiltonian.integrals, device=str(d)))
    return hams


# -----------------------------------------------------------------------------
def compute_coupling_multi_gpu(new_candidates, sci_state,
                                hamiltonians: List, n_orb, n_qubits):
    """Split candidates across K hamiltonians, parallel compute_coupling_gpu.

    Each candidate's coupling is independent; no cross-terms. Safe to split.
    """
    K = len(hamiltonians)
    N = len(new_candidates)
    if N == 0:
        return np.zeros(0)
    if K == 1:
        return compute_coupling_gpu(new_candidates, sci_state, hamiltonians[0],
                                     n_orb, n_qubits)

    chunks = [new_candidates[i * N // K:(i + 1) * N // K] for i in range(K)]

    def _work(chunk, ham, gpu_idx):
        if not chunk:
            return (gpu_idx, np.zeros(0))
        torch.cuda.set_device(gpu_idx)
        return (gpu_idx, compute_coupling_gpu(chunk, sci_state, ham, n_orb, n_qubits))

    with concurrent.futures.ThreadPoolExecutor(max_workers=K) as exe:
        futures = [exe.submit(_work, chunks[i], hamiltonians[i], i) for i in range(K)]
        results = [f.result() for f in futures]

    # Re-assemble in original order
    results.sort(key=lambda x: x[0])
    return np.concatenate([r[1] for r in results])


# -----------------------------------------------------------------------------
def final_pt2_multi_gpu(cumulative_bs, sci_state, hamiltonians: List,
                        n_orb, n_qubits, e0, pt2_top_n, chunk):
    """Multi-GPU final PT2.

    Phase 1 (single master GPU): enumerate externals from top-N amp dets, dedup.
    Phase 2 (K GPUs): split unique externals across GPUs, compute coupling + PT2.
    Reduce.

    Returns (E_PT2, n_externals).
    """
    K = len(hamiltonians)
    master = hamiltonians[0]

    amps = np.asarray(sci_state.amplitudes)
    abs_flat = np.abs(amps).flatten()
    nonzero = int((abs_flat > 1e-14).sum())
    if nonzero == 0:
        return 0.0, 0
    top_n = min(pt2_top_n, nonzero)

    top_flat = np.argpartition(-abs_flat, top_n - 1)[:top_n]
    n_b = amps.shape[1]
    top_ia = top_flat // n_b
    top_ib = top_flat % n_b
    ci_strs_a = np.asarray(sci_state.ci_strs_a)
    ci_strs_b = np.asarray(sci_state.ci_strs_b)

    configs_np = np.zeros((top_n, n_qubits), dtype=np.int64)
    for i in range(top_n):
        a = int(ci_strs_a[top_ia[i]])
        b = int(ci_strs_b[top_ib[i]])
        for k in range(n_orb):
            configs_np[i, k] = (a >> k) & 1
            configs_np[i, n_orb + k] = (b >> k) & 1
    configs_t = torch.from_numpy(configs_np).long()

    basis_hashes = {cumulative_bs[i].tobytes() for i in range(len(cumulative_bs))}

    # ----- Phase 1: enumerate + dedup externals on master GPU -----
    seen_external = set(basis_hashes)
    ext_rows = []       # list of ibm row arrays
    ext_hashes = []
    ext_configs = []    # list of NQS-format config tensors (cpu)

    for start in range(0, top_n, chunk):
        end = min(start + chunk, top_n)
        batch = configs_t[start:end].to(master.device)
        try:
            connected, _, _ = master.get_connections_vectorized_batch(batch)
        except (MemoryError, RuntimeError):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        if len(connected) == 0:
            continue
        conn_unique = torch.unique(connected.long().cpu(), dim=0)
        conn_ibm = _configs_to_ibm_format(conn_unique, n_orb, n_qubits)
        for i in range(len(conn_ibm)):
            h = conn_ibm[i].tobytes()
            if h not in seen_external:
                seen_external.add(h)
                ext_rows.append(conn_ibm[i])
                ext_hashes.append(h)
                ext_configs.append(conn_unique[i])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    N_ext = len(ext_rows)
    if N_ext == 0:
        return 0.0, 0

    # ----- Phase 2: split externals across K GPUs; compute coupling + PT2 -----
    def _work(slice_ext, ham, gpu_idx):
        if not slice_ext:
            return 0.0, 0
        torch.cuda.set_device(gpu_idx)
        coupling = compute_coupling_gpu(slice_ext, sci_state, ham, n_orb, n_qubits)
        cfgs = torch.stack([c[2] for c in slice_ext]).to(ham.device)
        h_diag = ham.diagonal_elements_batch(cfgs).cpu().numpy()
        denom = e0 - h_diag
        safe = np.abs(denom) > 1e-12
        e_pt2_part = float(np.sum((coupling[safe] ** 2) / denom[safe]))
        return e_pt2_part, int(safe.sum())

    # Build (row, hash, config) tuples expected by compute_coupling_gpu
    ext_list = list(zip(ext_rows, ext_hashes, ext_configs))

    if K == 1:
        return _work(ext_list, master, 0)

    slices = [ext_list[i * N_ext // K:(i + 1) * N_ext // K] for i in range(K)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=K) as exe:
        futures = [exe.submit(_work, slices[i], hamiltonians[i], i) for i in range(K)]
        results = [f.result() for f in futures]

    e_pt2 = sum(r[0] for r in results)
    n_ext = sum(r[1] for r in results)
    return e_pt2, n_ext
