"""GPU-native sparse determinant CI build + solve.

The whole hot path stays on GPU:
  1. Connection enumeration via hamiltonian.get_connections_vectorized_batch (GPU).
  2. "Is connection in basis?" lookup via torch.searchsorted (GPU).
  3. Sparse H stored as torch sparse COO on GPU.
  4. Davidson via gpu_davidson (torch.sparse.mm + torch.linalg.eigh) — GPU.

No scipy, no numpy conversion on the hot path. The only CPU touches are the
returned eigenvector (numpy for compat with downstream sci_state) and the
amplitudes lookup table (dict, cheap).

Expected speedup: 20-50x on N_det = 100k-500k basis vs pure-Python build + scipy eigsh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _encode_keys(configs: torch.Tensor, n_orb: int, powers: torch.Tensor):
    """Encode (alpha_int, beta_int) pair → single int64 key.

    configs: (N, n_qubits) torch.long, NQS format (bit k = orbital k occupied).
    Assumes n_orb <= 31 so a_int and b_int fit in 32 bits each.
    """
    a_ints = (configs[:, :n_orb].long() * powers).sum(dim=1)
    b_ints = (configs[:, n_orb:].long() * powers).sum(dim=1)
    MAX32 = 1 << 32
    return a_ints * MAX32 + b_ints  # (N,) int64


def build_sparse_hamiltonian_gpu(
    basis_configs: torch.Tensor,
    hamiltonian,
    ecore: float = 0.0,
) -> torch.Tensor:
    """Build sparse H matrix (N_det x N_det) on GPU; returns torch sparse COO.

    basis_configs: (N, n_qubits) torch.long on hamiltonian.device.
    """
    device = hamiltonian.device
    n_orb = hamiltonian.n_orbitals
    n_qubits = 2 * n_orb
    N = basis_configs.shape[0]
    if N == 0:
        empty_idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_val = torch.zeros(0, dtype=torch.float64, device=device)
        return torch.sparse_coo_tensor(empty_idx, empty_val, (0, 0)).coalesce()

    powers = (1 << torch.arange(n_orb, device=device, dtype=torch.long))

    # Encode basis → sorted keys for searchsorted lookup
    basis_keys = _encode_keys(basis_configs, n_orb, powers)  # (N,)
    sort_idx = basis_keys.argsort()
    sorted_keys = basis_keys[sort_idx]

    # Enumerate all connections of all basis dets (GPU batched).
    # Process in chunks and bisect further on MemoryError. Each chunk is
    # constrained to max_chunk configs at a time; on OOM, halve and retry.
    def _enumerate_chunks(configs_slice, start_offset, max_chunk):
        """Recursively enumerate connections for a slice, halving chunk on OOM."""
        out_c, out_e, out_bi = [], [], []
        i = 0
        cur_chunk = max_chunk
        while i < len(configs_slice):
            end = min(i + cur_chunk, len(configs_slice))
            try:
                c_conn, c_elem, c_bi = hamiltonian.get_connections_vectorized_batch(
                    configs_slice[i:end]
                )
            except (MemoryError, RuntimeError) as ex:
                if "max_output" in str(ex) or "MemoryError" in str(type(ex).__name__):
                    if cur_chunk <= 1:
                        raise
                    cur_chunk = max(1, cur_chunk // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
            c_bi = c_bi + (start_offset + i)
            out_c.append(c_conn.cpu())   # move to CPU to free GPU mem
            out_e.append(c_elem.cpu())
            out_bi.append(c_bi.cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            i = end
        return out_c, out_e, out_bi

    # Start with reasonable chunk size based on n_orb: 100k-basis experiments
    # crash at 25k/chunk, so start smaller.
    initial_chunk = min(N, 8000)
    out_c, out_e, out_bi = _enumerate_chunks(basis_configs, 0, initial_chunk)
    if out_c:
        connected = torch.cat([c.to(device) for c in out_c], dim=0)
        elements = torch.cat([e.to(device) for e in out_e], dim=0)
        batch_idx = torch.cat([bi.to(device) for bi in out_bi], dim=0)
    else:
        connected = torch.zeros((0, n_qubits), dtype=torch.long, device=device)
        elements = torch.zeros(0, dtype=torch.float64, device=device)
        batch_idx = torch.zeros(0, dtype=torch.long, device=device)

    # Lookup connected in basis via searchsorted
    if len(connected) > 0:
        conn_keys = _encode_keys(connected, n_orb, powers)
        pos = torch.searchsorted(sorted_keys, conn_keys)
        pos_clamped = torch.clamp(pos, max=N - 1)
        valid = sorted_keys[pos_clamped] == conn_keys
        row_i = batch_idx[valid].long()
        col_j = sort_idx[pos_clamped[valid]].long()
        vals = elements[valid].double()
    else:
        row_i = torch.zeros(0, dtype=torch.long, device=device)
        col_j = torch.zeros(0, dtype=torch.long, device=device)
        vals = torch.zeros(0, dtype=torch.double, device=device)

    # Diagonal.
    # IMPORTANT: hamiltonian.diagonal_elements_batch returns <config|H|config> in
    # FULL molecular energy units (i.e. includes the constant offset stored in
    # integrals.nuclear_repulsion — which for CAS systems is really e_core, not
    # just e_nuc). The CPU sparse_det_solver builds diag as e1 + 0.5*e2 (no
    # constant offset) and expects the caller to add nuclear_repulsion after
    # diagonalisation. To match that convention here, subtract nuclear_repulsion
    # from the diagonal so the eigenvalue is the electronic energy, consistent
    # with CPU build + the `e0 = e + nuclear_repulsion` addition in the main loop.
    nucrep = float(hamiltonian.integrals.nuclear_repulsion)
    diag = hamiltonian.diagonal_elements_batch(basis_configs).double() + ecore - nucrep
    row_diag = torch.arange(N, device=device, dtype=torch.long)

    all_rows = torch.cat([row_i, row_diag])
    all_cols = torch.cat([col_j, row_diag])
    all_vals = torch.cat([vals, diag])

    # Assemble torch sparse COO on GPU — no CPU round-trip, no scipy.
    indices = torch.stack([all_rows, all_cols], dim=0)
    H = torch.sparse_coo_tensor(indices, all_vals, (N, N), dtype=torch.float64,
                                 device=device)
    return H.coalesce()


@dataclass
class GPUSparseDetResult:
    energy: float
    amplitudes: np.ndarray        # length N_det
    alpha_strs: np.ndarray
    beta_strs: np.ndarray


def solve_sparse_det_ci_gpu(
    alpha_strs: np.ndarray,
    beta_strs: np.ndarray,
    hamiltonian,
    ecore: float = 0.0,
    ci0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_cycle: int = 40,
    use_gpu_davidson: bool = True,         # GPU by default
    use_multi_gpu_davidson: bool = False,
    davidson_devices=None,
    use_on_the_fly: bool = True,           # ← DEFAULT on-the-fly matvec
    on_the_fly_chunk_size: int = 4000,
    use_multi_gpu_on_the_fly: bool = False,
    multi_gpu_hamiltonians=None,
) -> GPUSparseDetResult:
    """Build H on GPU, diagonalise on GPU.

    All operations stay on GPU: H build, Davidson matvec, subspace eigen.
    No scipy, no eigsh fall-back. If GPU Davidson fails, the error surfaces.

    alpha_strs, beta_strs: int64 arrays, length N_det, defining (alpha, beta) pairs.
    """
    device = hamiltonian.device
    n_orb = hamiltonian.n_orbitals
    n_qubits = 2 * n_orb
    N = len(alpha_strs)
    if N == 0:
        raise ValueError("Empty determinant basis")

    # Rebuild NQS-format configs from (alpha_int, beta_int) directly on GPU.
    a_t = torch.as_tensor(np.asarray(alpha_strs, dtype=np.int64)).to(device)
    b_t = torch.as_tensor(np.asarray(beta_strs, dtype=np.int64)).to(device)
    k_range = torch.arange(n_orb, device=device, dtype=torch.long)
    configs_t = torch.zeros((N, n_qubits), dtype=torch.long, device=device)
    configs_t[:, :n_orb] = ((a_t.unsqueeze(1) >> k_range.unsqueeze(0)) & 1)
    configs_t[:, n_orb:] = ((b_t.unsqueeze(1) >> k_range.unsqueeze(0)) & 1)

    # Multi-GPU on-the-fly: row-partition basis across K GPUs, parallel matvec.
    # Required for huge basis (1M-5M) where single-GPU enumeration is too slow.
    if use_multi_gpu_on_the_fly and multi_gpu_hamiltonians is not None \
            and len(multi_gpu_hamiltonians) > 1 and N > 1:
        from ._gpu_davidson import davidson_lowest_multi_gpu_on_the_fly
        v0 = None
        if ci0 is not None:
            v0 = np.asarray(ci0, dtype=np.float64).ravel()
            if v0.shape[0] == N:
                nv = float(np.linalg.norm(v0))
                v0 = v0 / nv if nv > 1e-12 else None
            else:
                v0 = None
        theta, eigvec_np = davidson_lowest_multi_gpu_on_the_fly(
            configs_t, multi_gpu_hamiltonians, ecore=ecore,
            v0=v0, tol=tol, max_iter=max_cycle * 2,
            chunk_size=on_the_fly_chunk_size,
        )
        return GPUSparseDetResult(
            energy=float(theta),
            amplitudes=np.asarray(eigvec_np, dtype=np.float64),
            alpha_strs=np.asarray(alpha_strs, dtype=np.int64),
            beta_strs=np.asarray(beta_strs, dtype=np.int64),
        )

    # Fast path: single-GPU on-the-fly matvec Davidson.
    if use_on_the_fly and not use_multi_gpu_davidson and N > 1:
        from ._gpu_davidson import davidson_lowest_on_the_fly
        v0 = None
        if ci0 is not None:
            v0 = np.asarray(ci0, dtype=np.float64).ravel()
            if v0.shape[0] == N:
                nv = float(np.linalg.norm(v0))
                v0 = v0 / nv if nv > 1e-12 else None
            else:
                v0 = None
        theta, eigvec_np = davidson_lowest_on_the_fly(
            configs_t, hamiltonian, ecore=ecore,
            v0=v0, tol=tol, max_iter=max_cycle * 2,
            chunk_size=on_the_fly_chunk_size,
        )
        return GPUSparseDetResult(
            energy=float(theta),
            amplitudes=np.asarray(eigvec_np, dtype=np.float64),
            alpha_strs=np.asarray(alpha_strs, dtype=np.int64),
            beta_strs=np.asarray(beta_strs, dtype=np.int64),
        )

    H = build_sparse_hamiltonian_gpu(configs_t, hamiltonian, ecore=ecore)

    if N == 1:
        # For N=1 just return the diagonal value (GPU → scalar)
        diag_val = float(H.to_dense()[0, 0].item())
        return GPUSparseDetResult(
            energy=diag_val,
            amplitudes=np.array([1.0], dtype=np.float64),
            alpha_strs=np.asarray(alpha_strs, dtype=np.int64),
            beta_strs=np.asarray(beta_strs, dtype=np.int64),
        )

    v0 = None
    if ci0 is not None:
        v0 = np.asarray(ci0, dtype=np.float64).ravel()
        if v0.shape[0] != N:
            v0 = None
        else:
            nv = float(np.linalg.norm(v0))
            v0 = v0 / nv if nv > 1e-12 else None

    # GPU Davidson (single or multi-GPU). No scipy fall-back.
    if use_multi_gpu_davidson:
        from ._multi_gpu_davidson import multi_gpu_davidson_lowest
        if davidson_devices is None:
            n_gpu = torch.cuda.device_count()
            davidson_devices = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
        # Multi-GPU path still expects a row-partitioned H; build it from torch H.
        # For simplicity keep H on master device; matvec is then device-local.
        theta, eigvec_np = _multi_gpu_dispatch(
            H, davidson_devices, v0=v0, tol=tol, max_iter=max_cycle * 2
        )
    else:
        from ._gpu_davidson import davidson_lowest
        diag_gpu = H.to_dense().diagonal()  # small cost vs total; keeps diag on GPU
        v0_t = None
        if v0 is not None:
            v0_t = torch.from_numpy(v0).to(device).double()
        theta, eigvec_t = davidson_lowest(
            H, diag_gpu, v0=v0_t, tol=tol, max_iter=max_cycle * 2,
        )
        eigvec_np = eigvec_t.detach().cpu().numpy()

    return GPUSparseDetResult(
        energy=float(theta),
        amplitudes=np.asarray(eigvec_np, dtype=np.float64),
        alpha_strs=np.asarray(alpha_strs, dtype=np.int64),
        beta_strs=np.asarray(beta_strs, dtype=np.int64),
    )


def _multi_gpu_dispatch(H_sparse_gpu: torch.Tensor, devices, v0, tol, max_iter):
    """Convert GPU torch sparse -> scipy CSR -> multi_gpu_davidson row-partition.

    This is a bridge for now; full GPU-native multi-GPU Davidson would need
    row-partitioned sparse tensors across devices, which multi_gpu_davidson.py
    does via scipy CSR as input (one-time per Davidson solve, not per matvec).
    """
    from scipy.sparse import csr_matrix
    from ._multi_gpu_davidson import multi_gpu_davidson_lowest
    H_coo = H_sparse_gpu.coalesce()
    indices = H_coo.indices().cpu().numpy()
    values = H_coo.values().cpu().numpy()
    N = H_sparse_gpu.shape[0]
    H_csr = csr_matrix((values, (indices[0], indices[1])), shape=(N, N))
    return multi_gpu_davidson_lowest(H_csr, devices, v0=v0, tol=tol, max_iter=max_iter)
