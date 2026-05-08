"""GPU-accelerated sparse det SQD backend. Drop-in for SparseDetSQDBackend."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._incremental_sqd import _SCIStateLike
from ._gpu_sparse_det_solver import solve_sparse_det_ci_gpu
from ._sparse_det_backend import _cumbs_to_alpha_beta


class GPUSparseDetSQDBackend:
    """GPU build + scipy eigsh diagonalisation.

    Requires a hamiltonian object (not just h1e/eri) because we reuse its
    get_connections_vectorized_batch for fast GPU enumeration.
    """

    def __init__(self, hamiltonian, n_alpha: int, n_beta: int,
                 spin_sq: Optional[float] = 0,
                 use_gpu_davidson: bool = False,
                 use_multi_gpu_davidson: bool = False,
                 davidson_devices=None,
                 use_multi_gpu_on_the_fly: bool = False,
                 multi_gpu_hamiltonians=None):
        self.hamiltonian = hamiltonian
        self.norb = hamiltonian.n_orbitals
        self.n_alpha = int(n_alpha)
        self.n_beta = int(n_beta)
        self.use_gpu_davidson = use_gpu_davidson
        self.use_multi_gpu_davidson = use_multi_gpu_davidson
        self.davidson_devices = davidson_devices
        self.use_multi_gpu_on_the_fly = use_multi_gpu_on_the_fly
        self.multi_gpu_hamiltonians = multi_gpu_hamiltonians
        self._prev_amps: dict = {}

    def solve(self, bitstring_matrix: np.ndarray):
        alpha_strs, beta_strs = _cumbs_to_alpha_beta(bitstring_matrix, self.norb)
        N = len(alpha_strs)
        if N == 0:
            raise ValueError("Empty bitstring_matrix passed to GPUSparseDetSQDBackend")

        # Warm-start ci0 from cached amplitudes
        ci0 = None
        if self._prev_amps:
            ci0 = np.fromiter(
                (self._prev_amps.get((int(a), int(b)), 0.0)
                 for a, b in zip(alpha_strs, beta_strs)),
                dtype=np.float64, count=N,
            )
            nrm = float(np.linalg.norm(ci0))
            if nrm < 1e-10:
                ci0 = None

        result = solve_sparse_det_ci_gpu(
            alpha_strs=alpha_strs, beta_strs=beta_strs,
            hamiltonian=self.hamiltonian, ecore=0.0, ci0=ci0,
            use_gpu_davidson=self.use_gpu_davidson,
            use_multi_gpu_davidson=self.use_multi_gpu_davidson,
            davidson_devices=self.davidson_devices,
            use_multi_gpu_on_the_fly=self.use_multi_gpu_on_the_fly,
            multi_gpu_hamiltonians=self.multi_gpu_hamiltonians,
        )

        self._prev_amps = {
            (int(a), int(b)): float(amp)
            for a, b, amp in zip(alpha_strs, beta_strs, result.amplitudes)
        }

        # Rebuild 2-D amplitude matrix for compat with _update_nqs
        unique_a = np.unique(alpha_strs)
        unique_b = np.unique(beta_strs)
        a_to_ia = {int(s): i for i, s in enumerate(unique_a)}
        b_to_ib = {int(s): i for i, s in enumerate(unique_b)}
        amps_2d = np.zeros((len(unique_a), len(unique_b)), dtype=np.float64)
        for k in range(N):
            ia = a_to_ia[int(alpha_strs[k])]
            ib = b_to_ib[int(beta_strs[k])]
            amps_2d[ia, ib] = result.amplitudes[k]

        sci_state = _SCIStateLike(
            amplitudes=amps_2d,
            ci_strs_a=unique_a.astype(np.int64),
            ci_strs_b=unique_b.astype(np.int64),
        )
        return float(result.energy), sci_state
