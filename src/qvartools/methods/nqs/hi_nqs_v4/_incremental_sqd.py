"""
Incremental SQD backend — bypasses IBM's solve_fermion to call PySCF's
kernel_fixed_space directly with persistent myci object and warm-start.

Why bypass solve_fermion?
  IBM's solve_fermion is designed for one-shot diagonalization. Per call it:
    1. Creates a fresh SelectedCI object (loses internal caches)
    2. Computes 1-RDM, 2-RDM, spin_square (expensive, unused by HI-NQS)
    3. Recomputes energy from RDMs instead of using Davidson eigenvalue
    4. Without proper SCIvector ci0, Davidson restarts cold

For iterative HI-NQS we want HCI-style behavior:
    - myci object persists across iterations
    - Skip RDM/spin computation (we only need amplitudes)
    - Warm-start Davidson from previous eigenvector via SCIvector ci0
"""
import numpy as np
from pyscf.fci import selected_ci, addons
from pyscf.fci.selected_ci import SCIvector, kernel_fixed_space
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs


class _SCIStateLike:
    """Minimal interface compatible with qiskit_addon_sqd.fermion.SCIState."""
    __slots__ = ("amplitudes", "ci_strs_a", "ci_strs_b")

    def __init__(self, amplitudes, ci_strs_a, ci_strs_b):
        self.amplitudes = amplitudes
        self.ci_strs_a = ci_strs_a
        self.ci_strs_b = ci_strs_b


class IncrementalSQDBackend:
    """
    Stateful SQD solver: persistent myci, warm-start, no RDM computation.

    Usage:
        backend = IncrementalSQDBackend(hcore, eri, n_alpha, n_beta, spin_sq=0)
        for iter in range(...):
            # build cumulative_bs ...
            e, sci_state = backend.solve(cumulative_bs)
    """

    def __init__(self, hcore, eri, n_alpha, n_beta, spin_sq=0):
        self.hcore = np.asarray(hcore)
        self.eri = np.asarray(eri)
        self.norb = self.hcore.shape[0]
        self.nelec = (n_alpha, n_beta)

        # Build SelectedCI ONCE (not per iteration)
        self.myci = selected_ci.SelectedCI()
        if spin_sq is not None:
            self.myci = addons.fix_spin_(self.myci, ss=spin_sq, shift=0.1)

        self.prev_sci_vec = None  # SCIvector with ._strs

    def solve(self, bitstring_matrix):
        """
        Diagonalize H projected onto the subspace defined by bitstring_matrix.

        Returns:
            (energy, sci_state_like) where sci_state_like has
            .amplitudes (na, nb), .ci_strs_a, .ci_strs_b
        """
        ci_strs = bitstring_matrix_to_ci_strs(bitstring_matrix, open_shell=False)
        new_a = np.asarray(ci_strs[0])
        new_b = np.asarray(ci_strs[1])
        ci_strs = (new_a, new_b)

        # Build warm-start ci0
        ci0 = self._build_warm_start(new_a, new_b)

        # Direct call to kernel_fixed_space — no RDM, no spin_square
        e, sci_vec = kernel_fixed_space(
            self.myci, self.hcore, self.eri, self.norb, self.nelec,
            ci_strs, ci0=ci0,
        )

        # Cache for next call
        self.prev_sci_vec = sci_vec

        # Build compat sci_state
        amps = np.asarray(sci_vec)
        if amps.ndim == 1:
            amps = amps.reshape(len(new_a), len(new_b))
        return float(e), _SCIStateLike(amps, new_a, new_b)

    def _build_warm_start(self, new_a, new_b):
        """Map prev_sci_vec onto new (na, nb) basis with zero padding.

        Vectorized via searchsorted + np.ix_ scatter (was O(N_old²) Python
        nested loop).
        """
        if self.prev_sci_vec is None:
            return None

        old_strs = getattr(self.prev_sci_vec, "_strs", None)
        if old_strs is None:
            return None

        old_a = np.asarray(old_strs[0], dtype=np.int64)
        old_b = np.asarray(old_strs[1], dtype=np.int64)
        old_amp = np.asarray(self.prev_sci_vec)
        if old_amp.ndim == 1:
            old_amp = old_amp.reshape(len(old_a), len(old_b))

        new_a_int = np.asarray(new_a, dtype=np.int64)
        new_b_int = np.asarray(new_b, dtype=np.int64)

        def _map(old_arr, new_arr):
            """For each old string, find its index in new_arr, -1 if absent."""
            if len(new_arr) == 0:
                return np.full(len(old_arr), -1, dtype=np.int64)
            sort_idx = np.argsort(new_arr, kind="stable")
            sorted_new = new_arr[sort_idx]
            pos = np.searchsorted(sorted_new, old_arr)
            pos_clipped = np.clip(pos, 0, len(sorted_new) - 1)
            matched = sorted_new[pos_clipped] == old_arr
            return np.where(matched, sort_idx[pos_clipped], -1)

        ia_new = _map(old_a, new_a_int)
        ib_new = _map(old_b, new_b_int)

        ci0_arr = np.zeros((len(new_a_int), len(new_b_int)), dtype=np.float64)
        ia_valid = ia_new >= 0
        ib_valid = ib_new >= 0
        if ia_valid.any() and ib_valid.any():
            old_ia_idx = np.where(ia_valid)[0]
            old_ib_idx = np.where(ib_valid)[0]
            sub_amp = old_amp[np.ix_(old_ia_idx, old_ib_idx)]
            ci0_arr[np.ix_(ia_new[old_ia_idx], ib_new[old_ib_idx])] = sub_amp

        norm = float(np.linalg.norm(ci0_arr))
        if norm < 1e-10:
            return None
        ci0_arr /= norm

        # Wrap as SCIvector so kernel_fixed_space accepts it
        ci0_sci = ci0_arr.view(SCIvector)
        ci0_sci._strs = (new_a, new_b)
        return ci0_sci
