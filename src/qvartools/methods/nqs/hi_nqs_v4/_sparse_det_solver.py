"""Determinant-level sparse CI solver.

True selected CI: diagonalises H projected onto the EXACT set of sampled
determinants (alpha, beta) pairs, without the alpha x beta Cartesian-product
blowup of pyscf.fci.selected_ci.kernel_fixed_space.

Davidson vector length = N_det (NOT n_a * n_b).
Memory ~ O(N_det x avg_connections) instead of O(n_a * n_b).

Slater-Condon rules (chemist notation (pq|rs)):
  Diagonal     :  e1 + 0.5 * e2  [via pyscf fci_slow occ-list formula]
  alpha single :  sign * [h[q,p] + sum_{k in occ_a, k!=p} ((qp|kk) - (qk|kp))
                                 + sum_{k in occ_b} (qp|kk)]
  beta  single :  same symmetrically
  aa/bb double :  sign * [(q1 p1 | q2 p2) - (q1 p2 | q2 p1)]
  ab   double :  sign * (qa pa | qb pb)
Signs use the canonical ordered-bit convention (popcount of set bits strictly
between the removed and created orbital positions), verified against pyscf
direct_spin1 full FCI on small systems.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from pyscf import ao2mo


def _bit_positions(s: int, norb: int) -> list:
    return [i for i in range(norb) if s & (1 << i)]


def _sign_single(s: int, i: int, a: int) -> int:
    """Sign of a_dag_a a_i |s> assuming bit i is set and bit a is unset in s.

    For an ordered second-quantised bit string the sign equals
    (-1)^(number of set bits strictly between min(i,a) and max(i,a)).
    """
    if i == a:
        return 1
    lo = min(i, a)
    hi = max(i, a)
    if hi - lo == 1:
        return 1
    mask = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
    return -1 if bin(s & mask).count("1") & 1 else 1


def _sign_double_same_spin(s: int, p1: int, p2: int, q1: int, q2: int) -> int:
    """Sign for same-spin double excitation (p1,p2) -> (q1,q2).

    Computed as the product of two sequential single-excitation signs, using
    the intermediate bit string after the first single.
    """
    s1 = _sign_single(s, p1, q1)
    s_mid = s ^ (1 << p1) ^ (1 << q1)
    s2 = _sign_single(s_mid, p2, q2)
    return s1 * s2


def build_sparse_hamiltonian(
    alpha_strs: np.ndarray,
    beta_strs: np.ndarray,
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    ecore: float = 0.0,
) -> csr_matrix:
    """Build the N_det x N_det sparse CI Hamiltonian in determinant basis.

    Parameters
    ----------
    alpha_strs, beta_strs : int64 arrays, length N_det
        (alpha_strs[i], beta_strs[i]) defines Slater determinant i.
    h1e : (norb, norb)
        One-electron integrals in MO basis.
    eri : chemist-notation 2e integrals, any compact/full layout.
    ecore : scalar
        Added to every diagonal element (nuclear repulsion + frozen core).

    Returns
    -------
    H : scipy.sparse.csr_matrix, shape (N_det, N_det), symmetric real.
    """
    alpha_strs = np.asarray(alpha_strs, dtype=np.int64)
    beta_strs = np.asarray(beta_strs, dtype=np.int64)
    N = len(alpha_strs)
    if N == 0:
        return csr_matrix((0, 0), dtype=np.float64)

    g = ao2mo.restore(1, np.asarray(eri, dtype=np.float64), norb)
    J = np.einsum("iijj->ij", g)
    K = np.einsum("ijji->ij", g)
    h_diag = np.diag(h1e)

    # Cache occupied / virtual orbital lists per determinant
    occ_a_list = [_bit_positions(int(a), norb) for a in alpha_strs]
    occ_b_list = [_bit_positions(int(b), norb) for b in beta_strs]
    all_orbs = set(range(norb))
    vir_a_list = [[o for o in range(norb) if o not in set(oa)] for oa in occ_a_list]
    vir_b_list = [[o for o in range(norb) if o not in set(ob)] for ob in occ_b_list]

    det_idx = {
        (int(alpha_strs[i]), int(beta_strs[i])): i for i in range(N)
    }

    rows = []
    cols = []
    data = []

    def _push(i, j, val):
        rows.append(i)
        cols.append(j)
        data.append(val)
        if i != j:
            rows.append(j)
            cols.append(i)
            data.append(val)

    for i in range(N):
        a_i = int(alpha_strs[i])
        b_i = int(beta_strs[i])
        occ_a = occ_a_list[i]
        occ_b = occ_b_list[i]
        vir_a = vir_a_list[i]
        vir_b = vir_b_list[i]
        oa = np.array(occ_a, dtype=np.int64)
        ob = np.array(occ_b, dtype=np.int64)

        # ---- Diagonal (fci_slow style) ----
        e1 = h_diag[oa].sum() + h_diag[ob].sum()
        e2 = (
            J[np.ix_(oa, oa)].sum()
            + J[np.ix_(oa, ob)].sum()
            + J[np.ix_(ob, oa)].sum()
            + J[np.ix_(ob, ob)].sum()
            - K[np.ix_(oa, oa)].sum()
            - K[np.ix_(ob, ob)].sum()
        )
        _push(i, i, float(e1 + 0.5 * e2 + ecore))

        # ---- alpha single excitation (beta unchanged) ----
        for p in occ_a:
            for q in vir_a:
                a_new = a_i ^ (1 << p) ^ (1 << q)
                j = det_idx.get((a_new, b_i))
                if j is None or j <= i:
                    continue
                sign = _sign_single(a_i, p, q)
                elem = h1e[q, p]
                for k in occ_a:
                    if k != p:
                        elem += g[q, p, k, k] - g[q, k, k, p]
                for k in occ_b:
                    elem += g[q, p, k, k]
                val = sign * elem
                if abs(val) > 1e-14:
                    _push(i, j, val)

        # ---- beta single excitation (alpha unchanged) ----
        for p in occ_b:
            for q in vir_b:
                b_new = b_i ^ (1 << p) ^ (1 << q)
                j = det_idx.get((a_i, b_new))
                if j is None or j <= i:
                    continue
                sign = _sign_single(b_i, p, q)
                elem = h1e[q, p]
                for k in occ_b:
                    if k != p:
                        elem += g[q, p, k, k] - g[q, k, k, p]
                for k in occ_a:
                    elem += g[q, p, k, k]
                val = sign * elem
                if abs(val) > 1e-14:
                    _push(i, j, val)

        # ---- alpha-alpha same-spin double ----
        n_oa = len(occ_a)
        n_va = len(vir_a)
        for pi in range(n_oa):
            p1 = occ_a[pi]
            for p2i in range(pi + 1, n_oa):
                p2 = occ_a[p2i]
                for qi in range(n_va):
                    q1 = vir_a[qi]
                    for q2i in range(qi + 1, n_va):
                        q2 = vir_a[q2i]
                        a_new = a_i ^ (1 << p1) ^ (1 << p2) ^ (1 << q1) ^ (1 << q2)
                        j = det_idx.get((a_new, b_i))
                        if j is None or j <= i:
                            continue
                        sign = _sign_double_same_spin(a_i, p1, p2, q1, q2)
                        val = sign * (g[q1, p1, q2, p2] - g[q1, p2, q2, p1])
                        if abs(val) > 1e-14:
                            _push(i, j, val)

        # ---- beta-beta same-spin double ----
        n_ob = len(occ_b)
        n_vb = len(vir_b)
        for pi in range(n_ob):
            p1 = occ_b[pi]
            for p2i in range(pi + 1, n_ob):
                p2 = occ_b[p2i]
                for qi in range(n_vb):
                    q1 = vir_b[qi]
                    for q2i in range(qi + 1, n_vb):
                        q2 = vir_b[q2i]
                        b_new = b_i ^ (1 << p1) ^ (1 << p2) ^ (1 << q1) ^ (1 << q2)
                        j = det_idx.get((a_i, b_new))
                        if j is None or j <= i:
                            continue
                        sign = _sign_double_same_spin(b_i, p1, p2, q1, q2)
                        val = sign * (g[q1, p1, q2, p2] - g[q1, p2, q2, p1])
                        if abs(val) > 1e-14:
                            _push(i, j, val)

        # ---- alpha-beta mixed double ----
        for pa in occ_a:
            for qa in vir_a:
                a_new = a_i ^ (1 << pa) ^ (1 << qa)
                sign_a = _sign_single(a_i, pa, qa)
                for pb in occ_b:
                    for qb in vir_b:
                        b_new = b_i ^ (1 << pb) ^ (1 << qb)
                        j = det_idx.get((a_new, b_new))
                        if j is None or j <= i:
                            continue
                        sign_b = _sign_single(b_i, pb, qb)
                        val = sign_a * sign_b * g[qa, pa, qb, pb]
                        if abs(val) > 1e-14:
                            _push(i, j, val)

    H = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)
    H.sum_duplicates()
    return H


@dataclass
class SparseDetResult:
    energy: float
    amplitudes: np.ndarray  # length N_det
    alpha_strs: np.ndarray
    beta_strs: np.ndarray


def solve_sparse_det_ci(
    alpha_strs: np.ndarray,
    beta_strs: np.ndarray,
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    ecore: float = 0.0,
    ci0: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    max_cycle: int = 50,
) -> SparseDetResult:
    """Diagonalise H projected onto the (alpha, beta) determinant basis.

    The Davidson vector length equals len(alpha_strs), i.e. the real number of
    selected determinants, NOT n_a * n_b.
    """
    alpha_strs = np.asarray(alpha_strs, dtype=np.int64)
    beta_strs = np.asarray(beta_strs, dtype=np.int64)

    H = build_sparse_hamiltonian(alpha_strs, beta_strs, h1e, eri, norb, ecore=ecore)
    N = H.shape[0]
    if N == 0:
        raise ValueError("Empty determinant basis")
    if N == 1:
        return SparseDetResult(
            energy=float(H.toarray()[0, 0]),
            amplitudes=np.array([1.0], dtype=np.float64),
            alpha_strs=alpha_strs,
            beta_strs=beta_strs,
        )

    v0 = None
    if ci0 is not None:
        v0 = np.asarray(ci0, dtype=np.float64).ravel()
        if v0.shape[0] != N:
            v0 = None
        else:
            nv0 = float(np.linalg.norm(v0))
            if nv0 > 1e-12:
                v0 = v0 / nv0
            else:
                v0 = None

    # eigsh with which='SA' → smallest algebraic eigenvalue
    # Use ncv >= 20 for tighter Lanczos subspace (avoid spurious convergence
    # in dense spectra). Lower tol when no warm vector (cold = need precision).
    ncv = min(N, max(20, 2 * 1 + 1))   # k + 2*k + 1 lower bound; here k=1
    if N > 50:
        ncv = min(N, max(20, ncv))
    actual_tol = tol if v0 is not None else min(tol, 1e-12)
    eigvals, eigvecs = eigsh(
        H, k=1, which="SA", v0=v0, tol=actual_tol,
        maxiter=max_cycle * 50, ncv=ncv,
    )
    return SparseDetResult(
        energy=float(eigvals[0]),
        amplitudes=np.asarray(eigvecs[:, 0], dtype=np.float64),
        alpha_strs=alpha_strs,
        beta_strs=beta_strs,
    )
