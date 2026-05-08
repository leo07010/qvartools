"""Backend wrapper that plugs sparse_det_solver into the HI+NQS+SQD loop.

Exposes the same (energy, sci_state) interface as IncrementalSQDBackend, so it
can be dropped into run_hi_nqs_sqd with minimal changes to the main loop and to
_update_nqs (which consumes sci_state.amplitudes as a 2-D (n_a, n_b) array).

Key difference from IncrementalSQDBackend: Davidson operates on an N_det-length
vector, not on the n_a * n_b Cartesian-product tensor. For ghost-padded regimes
(n_a * n_b >> N_det) this is orders of magnitude cheaper.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._incremental_sqd import _SCIStateLike
from ._sparse_det_solver import solve_sparse_det_ci


def _cumbs_to_alpha_beta(bitstring_matrix: np.ndarray, n_orb: int):
    """Extract (alpha_int, beta_int) arrays from an IBM-format bitstring matrix.

    Uses the same MSB-first packing as _update_nqs in hi_nqs_sqd.py, which is
    consistent with pyscf cistring convention: bit k of the integer equals
    orbital k occupancy.
    """
    powers_msb = (1 << np.arange(n_orb - 1, -1, -1)).astype(np.int64)
    bs_int = np.asarray(bitstring_matrix).astype(np.int64)
    alpha_strs = (bs_int[:, :n_orb] * powers_msb).sum(axis=1)
    beta_strs = (bs_int[:, n_orb:] * powers_msb).sum(axis=1)
    return alpha_strs.astype(np.int64), beta_strs.astype(np.int64)


class SparseDetSQDBackend:
    """Determinant-level sparse CI backend.

    Davidson vector length = N_det (not n_a * n_b).

    Usage:
        backend = SparseDetSQDBackend(hcore, eri, n_alpha, n_beta)
        for iter in range(...):
            e, sci_state = backend.solve(cumulative_bs)
    """

    def __init__(self, hcore: np.ndarray, eri: np.ndarray,
                 n_alpha: int, n_beta: int, spin_sq: Optional[float] = 0):
        self.hcore = np.asarray(hcore, dtype=np.float64)
        self.eri = np.asarray(eri, dtype=np.float64)
        self.norb = self.hcore.shape[0]
        self.n_alpha = int(n_alpha)
        self.n_beta = int(n_beta)
        # Cache of amplitudes keyed by (alpha_int, beta_int) for warm-start
        self._prev_amps: dict = {}

    def solve(self, bitstring_matrix: np.ndarray):
        alpha_strs, beta_strs = _cumbs_to_alpha_beta(bitstring_matrix, self.norb)
        N = len(alpha_strs)
        if N == 0:
            raise ValueError("Empty bitstring_matrix passed to SparseDetSQDBackend.solve")

        # Warm-start: look up previous amplitudes for each current det
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

        result = solve_sparse_det_ci(
            alpha_strs=alpha_strs,
            beta_strs=beta_strs,
            h1e=self.hcore,
            eri=self.eri,
            norb=self.norb,
            ecore=0.0,       # nuclear_repulsion added by caller (same as incremental)
            ci0=ci0,
        )

        # Cache amplitudes for next warm-start
        self._prev_amps = {
            (int(a), int(b)): float(amp)
            for a, b, amp in zip(alpha_strs, beta_strs, result.amplitudes)
        }

        # Rebuild a 2-D (n_a, n_b) amplitude matrix so _update_nqs's alpha/beta
        # marginal extraction keeps working unchanged. Positions corresponding to
        # (alpha, beta) pairs NOT in the selected determinant basis are left at
        # zero — this is the correct sparse projection (no ghost padding).
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
