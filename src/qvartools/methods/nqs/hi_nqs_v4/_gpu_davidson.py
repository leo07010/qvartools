"""GPU-native Davidson with three matvec strategies.

Strategy A — materialized sparse H (legacy):
  H is built once as torch sparse COO on GPU; uses torch.sparse.mm.
  Memory: O(N_det × avg_connections). OOMs for 52Q basis ≥ 200k.

Strategy B — single-GPU on-the-fly matvec (default in v4):
  Never materialises H. Each matvec enumerates basis connections via
  hamiltonian.get_connections_vectorized_batch on chunks and uses
  torch.searchsorted + scatter_add to accumulate y = H v.
  Memory: O(chunk_size × avg_connections), a few GB for any basis size.
  Speed: 3-50× faster than A on small/medium basis (avoids H build).

Strategy C — multi-GPU on-the-fly matvec (new):
  Same as B but row-partitions basis across K visible GPUs. Each GPU
  enumerates only its slice of basis connections per matvec; partial y
  vectors are gathered on master.
  Memory per GPU: O(N/K + chunk×avg_conn). 1M-5M basis becomes feasible.
  Speed: ≈ B / K, near-linear scaling for matvec.
"""
from __future__ import annotations

import concurrent.futures
from typing import Callable, List, Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix


def _scipy_csr_to_torch_sparse(H_csr: csr_matrix, device) -> torch.Tensor:
    H_coo = H_csr.tocoo()
    indices_np = np.stack([H_coo.row, H_coo.col]).astype(np.int64)
    values_np = H_coo.data.astype(np.float64)
    indices = torch.from_numpy(indices_np).to(device)
    values = torch.from_numpy(values_np).to(device)
    H = torch.sparse_coo_tensor(indices, values, H_csr.shape, dtype=torch.float64)
    return H.coalesce()


def _encode_keys(configs: torch.Tensor, n_orb: int, powers: torch.Tensor):
    a_ints = (configs[:, :n_orb].long() * powers).sum(dim=1)
    b_ints = (configs[:, n_orb:].long() * powers).sum(dim=1)
    MAX32 = 1 << 32
    return a_ints * MAX32 + b_ints


# =============================================================================
# Strategy B: single-GPU on-the-fly
# =============================================================================
class OnTheFlyH:
    """On-the-fly H matvec operator on a single GPU. No sparse H stored."""

    def __init__(self, basis_configs: torch.Tensor, hamiltonian,
                 ecore: float = 0.0, chunk_size: int = 4000):
        self.basis_configs = basis_configs
        self.hamiltonian = hamiltonian
        self.n_orb = hamiltonian.n_orbitals
        self.device = hamiltonian.device
        self.N = basis_configs.shape[0]
        self.chunk_size = chunk_size

        self.powers = (1 << torch.arange(self.n_orb, device=self.device,
                                          dtype=torch.long))
        basis_keys = _encode_keys(basis_configs, self.n_orb, self.powers)
        self.sort_idx = basis_keys.argsort()
        self.sorted_keys = basis_keys[self.sort_idx]

        nucrep = float(hamiltonian.integrals.nuclear_repulsion)
        self.diag = (hamiltonian.diagonal_elements_batch(basis_configs).double()
                     + ecore - nucrep)
        self.shape = (self.N, self.N)

    @torch.no_grad()
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        y = self.diag * v
        N = self.N
        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)
            try:
                connected, elements, batch_idx = \
                    self.hamiltonian.get_connections_vectorized_batch(
                        self.basis_configs[i:end]
                    )
            except (MemoryError, RuntimeError):
                if self.chunk_size <= 1:
                    raise
                self.chunk_size = max(1, self.chunk_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self.matvec(v)

            if len(connected) == 0:
                continue
            conn_keys = _encode_keys(connected, self.n_orb, self.powers)
            pos = torch.searchsorted(self.sorted_keys, conn_keys)
            pos_clamped = torch.clamp(pos, max=N - 1)
            valid = self.sorted_keys[pos_clamped] == conn_keys

            col_j = self.sort_idx[pos_clamped[valid]]
            row_i = batch_idx[valid].long() + i
            vals = elements[valid].double()

            y.scatter_add_(0, row_i, vals * v[col_j])

        return y


# =============================================================================
# Strategy C: multi-GPU on-the-fly
# =============================================================================
class MultiGPUOnTheFlyH:
    """Multi-GPU on-the-fly matvec. Row-partitions basis across K GPUs.

    Each GPU holds:
      - its slice of basis_configs (rows [row_splits[k]:row_splits[k+1]])
      - sorted basis keys for searchsorted lookup (full array, replicated)
      - sort_idx for column-index recovery (full, replicated)
      - hamiltonian replica
      - powers tensor

    Matvec dispatches K parallel threads, each computes its partial y slice.
    Master gathers and assembles.
    """

    def __init__(self, basis_configs: torch.Tensor,
                 hamiltonians: List, ecore: float = 0.0,
                 chunk_size: int = 4000):
        self.hamiltonians = hamiltonians
        self.K = len(hamiltonians)
        self.devices = [h.device for h in hamiltonians]
        self.master = self.devices[0]
        self.n_orb = hamiltonians[0].n_orbitals
        self.N = basis_configs.shape[0]
        self.chunk_size = chunk_size

        # Compute sorted keys & sort_idx ONCE on master, then replicate to other GPUs.
        powers_master = (1 << torch.arange(self.n_orb, device=self.master,
                                            dtype=torch.long))
        basis_master = basis_configs.to(self.master)
        all_keys = _encode_keys(basis_master, self.n_orb, powers_master)
        sort_idx_master = all_keys.argsort()
        sorted_keys_master = all_keys[sort_idx_master]

        # Distribute basis (row-partition) and replicate sorted_keys/sort_idx.
        self.row_splits = [k * self.N // self.K for k in range(self.K + 1)]
        self.basis_per_gpu = []
        self.sorted_keys_per_gpu = []
        self.sort_idx_per_gpu = []
        self.powers_per_gpu = []
        for k in range(self.K):
            rs, re = self.row_splits[k], self.row_splits[k + 1]
            self.basis_per_gpu.append(basis_master[rs:re].to(self.devices[k]))
            self.sorted_keys_per_gpu.append(sorted_keys_master.to(self.devices[k]))
            self.sort_idx_per_gpu.append(sort_idx_master.to(self.devices[k]))
            self.powers_per_gpu.append(
                (1 << torch.arange(self.n_orb, device=self.devices[k],
                                   dtype=torch.long))
            )

        # Diagonal on master.
        nucrep = float(hamiltonians[0].integrals.nuclear_repulsion)
        self.diag = (hamiltonians[0].diagonal_elements_batch(basis_master).double()
                     + ecore - nucrep)
        self.shape = (self.N, self.N)

    @torch.no_grad()
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        v_master = v.to(self.master)
        y = self.diag * v_master  # diagonal contribution (master)

        v_per_gpu = [v_master.to(d, non_blocking=True) for d in self.devices]

        def _work(k):
            torch.cuda.set_device(self.devices[k])
            ham = self.hamiltonians[k]
            basis_k = self.basis_per_gpu[k]
            sorted_keys_k = self.sorted_keys_per_gpu[k]
            sort_idx_k = self.sort_idx_per_gpu[k]
            powers_k = self.powers_per_gpu[k]
            v_k = v_per_gpu[k]
            N_local = basis_k.shape[0]
            row_offset = self.row_splits[k]

            y_partial = torch.zeros(N_local, dtype=torch.float64,
                                     device=self.devices[k])
            cs = self.chunk_size
            for i in range(0, N_local, cs):
                end = min(i + cs, N_local)
                try:
                    connected, elements, batch_idx = \
                        ham.get_connections_vectorized_batch(basis_k[i:end])
                except (MemoryError, RuntimeError):
                    if cs <= 1:
                        raise
                    cs = max(1, cs // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                if len(connected) == 0:
                    continue
                conn_keys = _encode_keys(connected, self.n_orb, powers_k)
                pos = torch.searchsorted(sorted_keys_k, conn_keys)
                pos_clamped = torch.clamp(pos, max=self.N - 1)
                valid = sorted_keys_k[pos_clamped] == conn_keys
                col_j = sort_idx_k[pos_clamped[valid]]
                row_i_local = batch_idx[valid].long() + i
                vals = elements[valid].double()
                y_partial.scatter_add_(0, row_i_local,
                                        vals * v_k[col_j])
            return (k, y_partial.to(self.master, non_blocking=True))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.K) as exe:
            futures = [exe.submit(_work, k) for k in range(self.K)]
            results = [f.result() for f in futures]
        results.sort(key=lambda r: r[0])
        for k, y_part in results:
            rs, re = self.row_splits[k], self.row_splits[k + 1]
            y[rs:re] += y_part
        return y


# =============================================================================
# Davidson loop (works with any matvec callable)
# =============================================================================
def davidson_lowest_generic(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    diag: torch.Tensor,
    N: int,
    v0: Optional[torch.Tensor] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
    max_subspace: int = 20,
    device=None,
):
    if device is None:
        device = diag.device

    if v0 is None:
        i0 = int(diag.argmin().item())
        v = torch.zeros(N, dtype=torch.float64, device=device)
        v[i0] = 1.0
    else:
        v = torch.as_tensor(v0, dtype=torch.float64, device=device).clone()
        n = v.norm()
        if n < 1e-12:
            i0 = int(diag.argmin().item())
            v = torch.zeros(N, dtype=torch.float64, device=device)
            v[i0] = 1.0
        else:
            v = v / n

    V = v.unsqueeze(1)
    HV = matvec(v).unsqueeze(1)

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
        HV = torch.cat([HV, matvec(t).unsqueeze(1)], dim=1)

        if V.shape[1] >= max_subspace:
            un = u.norm()
            V = (u / un).unsqueeze(1) if un > 1e-12 else u.unsqueeze(1)
            HV = matvec(V.squeeze(1)).unsqueeze(1)

    return best_theta, best_u


def davidson_lowest(
    H_sp,
    diag: torch.Tensor,
    v0: Optional[torch.Tensor] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
    max_subspace: int = 20,
):
    if not isinstance(H_sp, torch.Tensor):
        H_sp = _scipy_csr_to_torch_sparse(H_sp, diag.device)
    N = H_sp.shape[0]
    device = H_sp.device

    def matvec(v):
        return torch.sparse.mm(H_sp, v.unsqueeze(1)).squeeze(1)

    return davidson_lowest_generic(
        matvec, diag, N, v0=v0, tol=tol, max_iter=max_iter,
        max_subspace=max_subspace, device=device,
    )


def davidson_lowest_on_the_fly(
    basis_configs: torch.Tensor,
    hamiltonian,
    ecore: float = 0.0,
    v0=None,
    tol: float = 1e-6,
    max_iter: int = 60,
    chunk_size: int = 4000,
):
    """Single-GPU on-the-fly Davidson."""
    op = OnTheFlyH(basis_configs, hamiltonian, ecore=ecore, chunk_size=chunk_size)
    v0_t = None
    if v0 is not None:
        v0_t = torch.as_tensor(v0, dtype=torch.float64, device=op.device)
    theta, u = davidson_lowest_generic(
        op.matvec, op.diag, op.N, v0=v0_t, tol=tol, max_iter=max_iter,
        device=op.device,
    )
    return theta, u.detach().cpu().numpy()


def davidson_lowest_multi_gpu_on_the_fly(
    basis_configs: torch.Tensor,
    hamiltonians: List,
    ecore: float = 0.0,
    v0=None,
    tol: float = 1e-6,
    max_iter: int = 60,
    chunk_size: int = 4000,
):
    """Multi-GPU on-the-fly Davidson. Distributes basis rows across GPUs."""
    op = MultiGPUOnTheFlyH(basis_configs, hamiltonians, ecore=ecore,
                            chunk_size=chunk_size)
    v0_t = None
    if v0 is not None:
        v0_t = torch.as_tensor(v0, dtype=torch.float64, device=op.master)
    theta, u = davidson_lowest_generic(
        op.matvec, op.diag, op.N, v0=v0_t, tol=tol, max_iter=max_iter,
        device=op.master,
    )
    return theta, u.detach().cpu().numpy()


def solve_lowest_gpu_davidson(
    H_csr,
    v0=None,
    tol: float = 1e-6,
    max_iter: int = 60,
    device=None,
):
    """Legacy entry: scipy CSR → GPU Davidson."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H_sp = _scipy_csr_to_torch_sparse(H_csr, device)
    diag_np = H_csr.diagonal().astype(np.float64)
    diag = torch.from_numpy(diag_np).to(device)
    v0_t = None
    if v0 is not None:
        v0_t = torch.as_tensor(v0, dtype=torch.float64, device=device)
    theta, u = davidson_lowest(H_sp, diag, v0=v0_t, tol=tol, max_iter=max_iter)
    return theta, u.detach().cpu().numpy()
