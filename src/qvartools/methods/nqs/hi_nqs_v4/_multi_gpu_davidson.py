"""Multi-GPU Davidson for lowest eigenvalue of sparse H.

Distributes H across K GPUs by rows. Each matvec y = H @ v is executed in
parallel: GPU k computes y[row_start_k : row_end_k] = H[row_start_k : row_end_k, :] @ v
(requires the full v on each GPU). Partial results are gathered on the master
device and concatenated.

Subspace operations (dot products, Gram-Schmidt, small subspace eigensolve) run
on the master GPU since the Krylov subspace is small (<= max_subspace ~20 cols).

For multi-GPU scaling, the compute-bound fraction is the matvec. Communication
is O(N) per matvec per GPU (tiny compared to sparse matvec compute). Near-linear
speedup expected up to K ~ nnz / (memory bandwidth × transfer latency) bound.

Fall-back: if K == 1, identical behaviour to single-GPU gpu_davidson.davidson_lowest.
"""
from __future__ import annotations

import concurrent.futures
from typing import List, Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix


def _csr_slice_to_torch_sparse(H_slice_csr: csr_matrix, device) -> torch.Tensor:
    """Convert scipy CSR row-slice to torch sparse COO on device (float64)."""
    H_coo = H_slice_csr.tocoo()
    indices = np.stack([H_coo.row, H_coo.col]).astype(np.int64)
    values = H_coo.data.astype(np.float64)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices).to(device),
        torch.from_numpy(values).to(device),
        H_slice_csr.shape, dtype=torch.float64,
    ).coalesce()


class MultiGPUSparseH:
    """Row-partitioned sparse H across K GPUs.

    Usage:
        mgh = MultiGPUSparseH(H_csr, devices)
        y = mgh.matvec(v)   # y = H @ v, v and y on devices[0] (master)
    """

    def __init__(self, H_csr: csr_matrix, devices: List[torch.device]):
        self.N = H_csr.shape[0]
        self.K = len(devices)
        self.devices = devices
        self.master = devices[0]
        self.row_splits = [k * self.N // self.K for k in range(self.K + 1)]
        self.H_local = []
        for k in range(self.K):
            rs, re = self.row_splits[k], self.row_splits[k + 1]
            H_slice = H_csr[rs:re]
            self.H_local.append(_csr_slice_to_torch_sparse(H_slice, devices[k]))

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Distributed y = H @ v. v on master device; returns y on master.

        Supports v shape (N,) or (N, k) for matrix-matrix.
        """
        squeeze = False
        if v.dim() == 1:
            v = v.unsqueeze(1)
            squeeze = True

        if self.K == 1:
            y = torch.sparse.mm(self.H_local[0], v)
            return y.squeeze(1) if squeeze else y

        # Parallel dispatch one thread per GPU
        def _work(k):
            v_k = v.to(self.devices[k], non_blocking=True)
            y_k = torch.sparse.mm(self.H_local[k], v_k)
            return y_k.to(self.master, non_blocking=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.K) as exe:
            futures = [exe.submit(_work, k) for k in range(self.K)]
            parts = [f.result() for f in futures]

        y = torch.cat(parts, dim=0)
        return y.squeeze(1) if squeeze else y


def multi_gpu_davidson_lowest(
    H_csr: csr_matrix,
    devices: List[torch.device],
    v0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
    max_subspace: int = 20,
):
    """Multi-GPU Davidson on row-partitioned H_csr.

    Returns (eigenvalue_float, eigenvector_numpy).
    """
    master = devices[0]
    N = H_csr.shape[0]

    mgh = MultiGPUSparseH(H_csr, devices)

    diag_np = H_csr.diagonal().astype(np.float64)
    diag = torch.from_numpy(diag_np).to(master)

    if v0 is None:
        i0 = int(diag.argmin().item())
        v = torch.zeros(N, dtype=torch.float64, device=master)
        v[i0] = 1.0
    else:
        v = torch.as_tensor(v0, dtype=torch.float64, device=master).clone()
        nn = v.norm()
        if nn < 1e-12:
            i0 = int(diag.argmin().item())
            v = torch.zeros(N, dtype=torch.float64, device=master)
            v[i0] = 1.0
        else:
            v = v / nn

    V = v.unsqueeze(1)                  # (N, 1) on master
    HV = mgh.matvec(V)                  # (N, 1) on master

    prev_theta = float("inf")
    best_theta = float("inf")
    best_u = v

    for it in range(max_iter):
        T = V.T @ HV
        T = 0.5 * (T + T.T)
        eigvals, eigvecs = torch.linalg.eigh(T)
        theta = float(eigvals[0].item())
        y = eigvecs[:, 0]
        u = V @ y
        r = HV @ y - theta * u
        res_norm = float(r.norm().item())

        best_theta = theta
        best_u = u

        if res_norm < tol or abs(theta - prev_theta) < tol * 0.01:
            break
        prev_theta = theta

        denom = diag - theta
        denom = torch.where(denom.abs() < 1e-10, torch.full_like(denom, 1e-10), denom)
        t = r / denom

        for _ in range(2):
            coef = V.T @ t
            t = t - V @ coef

        tn = float(t.norm().item())
        if tn < 1e-12:
            break
        t = t / tn

        V = torch.cat([V, t.unsqueeze(1)], dim=1)
        HV = torch.cat([HV, mgh.matvec(t.unsqueeze(1))], dim=1)

        if V.shape[1] >= max_subspace:
            un = u.norm()
            V = (u / un).unsqueeze(1) if un > 1e-12 else u.unsqueeze(1)
            HV = mgh.matvec(V)

    return best_theta, best_u.detach().cpu().numpy()


def solve_lowest_multi_gpu_davidson(
    H_csr: csr_matrix,
    devices: Optional[List[torch.device]] = None,
    v0=None,
    tol: float = 1e-6,
    max_iter: int = 60,
):
    """Convenience wrapper. devices=None -> all visible CUDA GPUs."""
    if devices is None:
        n_gpu = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
    return multi_gpu_davidson_lowest(H_csr, devices, v0=v0, tol=tol, max_iter=max_iter)
