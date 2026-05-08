"""
HI+NQS+SQD v3: NQS sampling + PT2 selection + SQD diagonalization.

Flow:
  1. NQS generates candidate configs (broad exploration)
  2. PT2 score ranks candidates (precision filtering)
  3. Top-k configs sent to solve_fermion (efficient diag)
  4. Eigenvector feeds back to NQS (targeted learning)

PT2 score: score(x) = |⟨x|H|Φ₀⟩|² / |E₀ - H_xx|
  - Iter 0: no Φ₀ yet → use diagonal energy ranking
  - Iter 1+: full PT2 using eigenvector from previous solve_fermion
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ._solver_result import SolverResult
from ._transformer import AutoregressiveTransformer

from qiskit_addon_sqd.fermion import solve_fermion, bitstring_matrix_to_ci_strs
from pyscf.fci.selected_ci import SCIvector
from ._incremental_sqd import IncrementalSQDBackend
from ._sparse_det_backend import SparseDetSQDBackend


@dataclass
class HINQSSQDConfig:
    max_iterations: int = 30
    convergence_threshold: float = 1e-6
    convergence_window: int = 3

    # NQS sampling
    n_samples: int = 5000

    # PT2 selection
    top_k: int = 2000           # keep top-k configs per iteration
    max_basis_size: int = 10000  # total basis cap (evict by PT2 score)

    # NQS update
    nqs_steps: int = 10
    nqs_lr: float = 1e-3

    # On-policy samples for energy gradient (0 = disabled)
    n_onpol_samples: int = 0

    # Loss weights
    teacher_weight: float = 1.0
    energy_weight: float = 0.0
    entropy_weight: float = 0.05

    # Temperature
    initial_temperature: float = 1.0
    final_temperature: float = 0.3

    # SQD warm-start (use previous eigenvector as initial guess)
    # CHANGED 2026-05-01: default False after discovering Davidson lock-in bug.
    # warm_start can cause Davidson to converge to wrong eigenvalue when iter k+1
    # basis adds dets weakly H-coupled to iter k eigenvector. Verified on
    # stretched N2 R=1.8: warm gave +110.6 mHa, cold gave +0.005 mHa on SAME basis.
    # Set to True only for single-reference systems where you trust convergence.
    warm_start: bool = False

    # Use incremental SQD backend (bypass solve_fermion, persist myci, skip RDM).
    # ~5-30x faster on large basis (mimics HCI's stateful solver).
    use_incremental_sqd: bool = True

    # Use true determinant-level sparse solver (overrides use_incremental_sqd).
    # Davidson vector length == N_det instead of n_a * n_b — recovers genuine
    # HCI-style cost on NQS-sampled bases where n_a * n_b >> N_det.
    use_sparse_det_solver: bool = False


    # Augmented teacher: extend supervision to H-connected dets outside basis.
    # aug_extension_weight=0 disables (standard joint-c² teacher only).
    # aug_extension_weight=0.5 means 50% of teacher mass is on extension dets.
    use_aug_teacher: bool = False
    aug_extension_weight: float = 0.5

    # Monotonic basis growth: never evict, never rescore existing basis.
    # When True:
    #   - Iter 0 uses cold_start_k (or top_k // 4) configs to limit cold-start noise
    #   - Iter 1+ only PT2-scores new candidates and APPENDS to basis
    #   - max_basis_size is ignored (always treated as 0)
    # This guarantees ci_strs(iter+1) ⊇ ci_strs(iter), giving Davidson a perfect
    # warm-start ci0 with no zero-padding loss. Trade-off: basis grows unbounded.
    monotonic_basis: bool = False
    cold_start_k: Optional[int] = None  # None → top_k // 4 if monotonic else top_k


def run_hi_nqs_sqd(hamiltonian, mol_info,
                   config: Optional[HINQSSQDConfig] = None) -> SolverResult:
    t0 = time.time()
    cfg = config or HINQSSQDConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    n_qubits = 2 * n_orb

    integrals = hamiltonian.integrals
    hcore = np.asarray(integrals.h1e, dtype=np.float64)
    eri = np.asarray(integrals.h2e, dtype=np.float64)
    nuclear_repulsion = float(integrals.nuclear_repulsion)

    # Auto-scale transformer
    if n_orb <= 5:
        embed, heads, layers = 64, 4, 3
    elif n_orb <= 7:
        embed, heads, layers = 128, 4, 4
    elif n_orb <= 10:
        embed, heads, layers = 128, 8, 6
    elif n_orb <= 15:
        embed, heads, layers = 192, 8, 6
    elif n_orb <= 20:
        embed, heads, layers = 256, 8, 8
    else:
        embed, heads, layers = 256, 8, 10

    nqs = AutoregressiveTransformer(
        n_orbitals=n_orb, n_alpha=n_alpha, n_beta=n_beta,
        embed_dim=embed, n_heads=heads, n_layers=layers,
    ).to(device)

    optimizer = torch.optim.Adam(nqs.parameters(), lr=cfg.nqs_lr)

    n_params = sum(p.numel() for p in nqs.parameters())
    print(f"    HI+NQS+SQD v3 (GPU={device}): arch={embed}/{heads}/{layers}, "
          f"params={n_params:,}, samples={cfg.n_samples}, top_k={cfg.top_k}")

    # State
    energy_history = []
    basis_size_history = []
    prev_energy = float("inf")
    best_energy = float("inf")
    converged = False
    converge_count = 0

    # Basis management
    cumulative_bs = None       # IBM format bool array
    cumulative_hashes = set()
    cumulative_scores = {}     # hash → PT2 score

    # Previous eigenvector info (for PT2 scoring from iter 1+)
    prev_sci_state = None
    current_e0 = None
    needs_rescore = False  # True after iter 0 → trigger full rescore at iter 1

    # SQD backend selection (priority: sparse_det > incremental > legacy solve_fermion)
    sqd_backend = None
    if cfg.use_sparse_det_solver:
        sqd_backend = SparseDetSQDBackend(
            hcore=hcore, eri=eri,
            n_alpha=n_alpha, n_beta=n_beta,
        )
    elif cfg.use_incremental_sqd:
        sqd_backend = IncrementalSQDBackend(
            hcore=hcore, eri=eri,
            n_alpha=n_alpha, n_beta=n_beta, spin_sq=0,
        )

    for iteration in range(cfg.max_iterations):
        iter_t0 = time.time()

        progress = iteration / max(cfg.max_iterations - 1, 1)
        temperature = (cfg.initial_temperature
                       + progress * (cfg.final_temperature - cfg.initial_temperature))

        # =====================================================
        # Step 1: NQS sampling
        # =====================================================
        sample_t0 = time.time()
        with torch.no_grad():
            configs_gpu, _ = nqs.sample(cfg.n_samples, temperature=temperature)
            configs_cpu = configs_gpu.long().cpu()

            # Particle filter
            alpha_counts = configs_cpu[:, :n_orb].sum(dim=1)
            beta_counts = configs_cpu[:, n_orb:].sum(dim=1)
            valid = (alpha_counts == n_alpha) & (beta_counts == n_beta)
            new_configs = configs_cpu[valid]

        n_sampled = len(new_configs)

        # Dedup against existing basis
        new_candidates = []
        if len(new_configs) > 0:
            new_unique = torch.unique(new_configs, dim=0)
            new_bs = _configs_to_ibm_format(new_unique, n_orb, n_qubits)
            for i in range(len(new_bs)):
                h = new_bs[i].tobytes()
                if h not in cumulative_hashes:
                    new_candidates.append((new_bs[i], h, new_unique[i]))

        sample_time = time.time() - sample_t0

        # =====================================================
        # Step 2: PT2 scoring + top-k selection
        # =====================================================
        score_t0 = time.time()
        n_evicted = 0

        if not new_candidates and not needs_rescore:
            n_selected = 0
        elif prev_sci_state is None or current_e0 is None:
            # Iter 0: no Φ₀ yet → rank by diagonal energy, take cold_start_k
            # (will be rescored with PT2 at Iter 1 unless monotonic_basis=True)
            diag_configs = torch.stack([c[2] for c in new_candidates])
            diag_e = hamiltonian.diagonal_elements_batch(diag_configs).cpu().numpy()

            if cfg.monotonic_basis:
                # Smaller cold-start set so noisy iter-0 picks don't permanently
                # bloat the basis (we cannot evict them later).
                _csk = cfg.cold_start_k if cfg.cold_start_k is not None else max(1, cfg.top_k // 4)
            else:
                _csk = cfg.cold_start_k if cfg.cold_start_k is not None else cfg.top_k
            keep_n = min(_csk, len(diag_e))
            top_idx = np.argsort(diag_e)[:keep_n]

            new_rows = np.stack([new_candidates[idx][0] for idx in top_idx])
            for idx in top_idx:
                cumulative_hashes.add(new_candidates[idx][1])
            if cumulative_bs is None:
                cumulative_bs = new_rows
            else:
                cumulative_bs = np.vstack([cumulative_bs, new_rows])
            n_selected = len(top_idx)
            needs_rescore = not cfg.monotonic_basis
        else:
            # Iter 1+: PT2 scoring using eigenvector from previous solve_fermion
            #
            # At Iter 1 (needs_rescore=True): rescore ALL existing basis + new
            # candidates together, keep top max_basis_size. This cleans out
            # the unscreened Iter 0 configs.
            #
            # At Iter 2+: only score new candidates, add top_k to basis.

            if needs_rescore and cumulative_bs is not None:
                # === Iter 1 special: rescore everything ===
                # Combine existing basis + new candidates into one pool
                all_bs = []
                all_hashes = []
                all_configs = []

                # Existing basis → convert back to config tensors
                existing_configs = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
                for i in range(len(cumulative_bs)):
                    all_bs.append(cumulative_bs[i])
                    all_hashes.append(cumulative_bs[i].tobytes())
                    all_configs.append(existing_configs[i])

                # New candidates
                for bs_row, h, config_tensor in new_candidates:
                    all_bs.append(bs_row)
                    all_hashes.append(h)
                    all_configs.append(config_tensor)

                # PT2 score everything
                all_as_candidates = [
                    (all_bs[i], all_hashes[i], all_configs[i])
                    for i in range(len(all_bs))
                ]
                diag_configs = torch.stack(all_configs)
                diag_e = hamiltonian.diagonal_elements_batch(diag_configs).cpu().numpy()

                coupling = _compute_coupling_to_ground_state(
                    all_as_candidates, prev_sci_state, hamiltonian,
                    n_orb, n_qubits,
                )

                scores = np.zeros(len(all_as_candidates))
                for i in range(len(all_as_candidates)):
                    denom = abs(current_e0 - diag_e[i])
                    if denom < 1e-12:
                        denom = 1e-12
                    scores[i] = coupling[i] ** 2 / denom

                # Keep top_k from the combined pool
                keep_n = min(cfg.top_k, len(scores))
                # Note: max_basis_size is NOT enforced here — Iter 2+ can grow freely
                top_idx = np.argsort(scores)[::-1][:keep_n]

                # Rebuild basis from scratch
                cumulative_bs = np.stack([all_bs[i] for i in top_idx])
                cumulative_hashes = {all_hashes[i] for i in top_idx}
                cumulative_scores = {all_hashes[i]: float(scores[i]) for i in top_idx}

                n_selected = len(top_idx) - len(existing_configs)  # net new
                n_evicted = len(existing_configs) + len(new_candidates) - len(top_idx)
                needs_rescore = False
            else:
                # === Iter 2+: only score new candidates ===
                if new_candidates:
                    diag_configs = torch.stack([c[2] for c in new_candidates])
                    diag_e = hamiltonian.diagonal_elements_batch(diag_configs).cpu().numpy()

                    coupling = _compute_coupling_to_ground_state(
                        new_candidates, prev_sci_state, hamiltonian,
                        n_orb, n_qubits,
                    )

                    scores = np.zeros(len(new_candidates))
                    for i in range(len(new_candidates)):
                        denom = abs(current_e0 - diag_e[i])
                        if denom < 1e-12:
                            denom = 1e-12
                        scores[i] = coupling[i] ** 2 / denom

                    top_idx = np.argsort(scores)[::-1][:cfg.top_k]

                    new_rows = np.stack([new_candidates[idx][0] for idx in top_idx])
                    for idx in top_idx:
                        h = new_candidates[idx][1]
                        cumulative_hashes.add(h)
                        cumulative_scores[h] = float(scores[idx])
                    cumulative_bs = np.vstack([cumulative_bs, new_rows])
                    n_selected = len(top_idx)
                else:
                    n_selected = 0

        # Seed with HF if empty
        if cumulative_bs is None:
            hf = hamiltonian.get_hf_state()
            hf_bs = _configs_to_ibm_format(hf.unsqueeze(0), n_orb, n_qubits)
            cumulative_bs = hf_bs
            cumulative_hashes.add(hf_bs[0].tobytes())

        # Evict lowest-scoring configs if over max_basis_size
        # (skipped entirely in monotonic_basis mode)
        if (not cfg.monotonic_basis
                and cfg.max_basis_size > 0
                and len(cumulative_bs) > cfg.max_basis_size):
            # Score all configs, keep top max_basis_size
            all_scores = np.array([
                cumulative_scores.get(cumulative_bs[i].tobytes(), 0.0)
                for i in range(len(cumulative_bs))
            ])
            keep_idx = np.argsort(all_scores)[::-1][:cfg.max_basis_size]
            keep_idx.sort()
            n_evicted = len(cumulative_bs) - len(keep_idx)
            cumulative_bs = cumulative_bs[keep_idx]
            cumulative_hashes = {row.tobytes() for row in cumulative_bs}

        score_time = time.time() - score_t0

        # =====================================================
        # Step 3: SQD diagonalization (single batch, full basis)
        # =====================================================
        sqd_t0 = time.time()

        try:
            if sqd_backend is not None:
                # Incremental backend: persistent myci, warm-start, skip RDM
                e, sci_state = sqd_backend.solve(cumulative_bs)
                e0 = e + nuclear_repulsion
            else:
                # Fallback: legacy solve_fermion path with optional ci0 warm-start
                ci0 = None
                if cfg.warm_start and prev_sci_state is not None:
                    try:
                        new_a, new_b = bitstring_matrix_to_ci_strs(
                            cumulative_bs, open_shell=False
                        )
                        new_a_arr = np.asarray(new_a)
                        new_b_arr = np.asarray(new_b)
                        a_lookup = {int(s): i for i, s in enumerate(new_a_arr)}
                        b_lookup = {int(s): i for i, s in enumerate(new_b_arr)}
                        ci0_arr = np.zeros(
                            (len(new_a_arr), len(new_b_arr)), dtype=np.float64
                        )
                        for ia, a_str in enumerate(prev_sci_state.ci_strs_a):
                            ia_new = a_lookup.get(int(a_str), -1)
                            if ia_new < 0:
                                continue
                            for ib, b_str in enumerate(prev_sci_state.ci_strs_b):
                                ib_new = b_lookup.get(int(b_str), -1)
                                if ib_new < 0:
                                    continue
                                ci0_arr[ia_new, ib_new] = (
                                    prev_sci_state.amplitudes[ia, ib]
                                )
                        norm = float(np.linalg.norm(ci0_arr))
                        if norm > 1e-10:
                            ci0_arr /= norm
                            ci0_sci = ci0_arr.view(SCIvector)
                            ci0_sci._strs = (new_a_arr, new_b_arr)
                            ci0 = ci0_sci
                    except Exception:
                        ci0 = None

                kwargs = {"ci0": ci0} if ci0 is not None else {}
                e, sci_state, occ, spin_sq = solve_fermion(
                    cumulative_bs, hcore, eri, spin_sq=0, **kwargs
                )
                e0 = e + nuclear_repulsion

            current_e0 = e0
            prev_sci_state = sci_state
        except Exception as ex:
            print(f"    Iter {iteration:>3d}: SQD failed: {ex}")
            continue

        sqd_time = time.time() - sqd_t0

        if e0 < best_energy:
            best_energy = e0

        energy_history.append(e0)
        basis_size_history.append(len(cumulative_bs))

        # =====================================================
        # Step 4: Update NQS with eigenvector teacher
        # =====================================================
        update_t0 = time.time()
        if getattr(cfg, 'use_aug_teacher', False):
            from ._nqs_aug_teacher_update import aug_teacher_update_nqs
            aug_teacher_update_nqs(
                nqs, optimizer, cumulative_bs, e0,
                sci_state, hamiltonian, cfg, device,
                n_orb, n_qubits,
            )
        else:
            _update_nqs(
                nqs, optimizer, cumulative_bs, e0,
                sci_state, hamiltonian, cfg, device,
                n_orb, n_qubits,
            )
        update_time = time.time() - update_t0

        # =====================================================
        # Step 5: Convergence
        # =====================================================
        delta_e = abs(e0 - prev_energy)
        prev_energy = e0
        iter_time = time.time() - iter_t0

        if delta_e < cfg.convergence_threshold and iteration > 0:
            converge_count += 1
        else:
            converge_count = 0

        print(f"    Iter {iteration:>3d}: E={e0:.10f}, "
              f"basis={len(cumulative_bs):>6d}(+{n_selected}, -{n_evicted}), "
              f"ΔE={delta_e:.2e}, "
              f"t={iter_time:.1f}s [samp={sample_time:.1f} pt2={score_time:.1f} "
              f"sqd={sqd_time:.1f} upd={update_time:.1f}]")

        if converge_count >= cfg.convergence_window:
            converged = True
            break

    wall_time = time.time() - t0

    return SolverResult(
        energy=best_energy if best_energy < float("inf") else None,
        diag_dim=len(cumulative_bs) if cumulative_bs is not None else 0,
        wall_time=wall_time,
        method="HI+NQS+SQD",
        converged=converged,
        metadata={
            "iterations": iteration + 1 if "iteration" in dir() else 0,
            "energy_history": energy_history,
            "basis_size_history": basis_size_history,
            "device": device,
        },
    )


def _build_coeff_lookup(sci_state, n_orb, device=None):
    """Build sorted-key tensors for O(log N) GPU-vectorized (a, b) → coeff lookup.

    Returns (sorted_keys, sorted_vals) — torch tensors on `device` if given,
    else CPU. Caller can pass these to `_lookup_coeffs_gpu` for in-place
    GPU lookup (avoids CPU↔GPU round-trip per chunk).
    """
    amps = np.asarray(sci_state.amplitudes)
    ci_strs_a = np.asarray(sci_state.ci_strs_a, dtype=np.int64)
    ci_strs_b = np.asarray(sci_state.ci_strs_b, dtype=np.int64)
    nonzero = np.argwhere(np.abs(amps) > 1e-14)
    if len(nonzero) == 0:
        empty_keys = torch.empty(0, dtype=torch.int64)
        empty_vals = torch.empty(0, dtype=torch.float64)
        if device is not None:
            empty_keys = empty_keys.to(device)
            empty_vals = empty_vals.to(device)
        return empty_keys, empty_vals
    a_keys = ci_strs_a[nonzero[:, 0]]
    b_keys = ci_strs_b[nonzero[:, 1]]
    keys = (a_keys.astype(np.int64) << n_orb) | b_keys.astype(np.int64)
    vals = amps[nonzero[:, 0], nonzero[:, 1]].astype(np.float64)
    order = np.argsort(keys)
    sorted_keys = torch.from_numpy(keys[order])
    sorted_vals = torch.from_numpy(vals[order])
    if device is not None:
        sorted_keys = sorted_keys.to(device)
        sorted_vals = sorted_vals.to(device)
    return sorted_keys, sorted_vals


def _lookup_coeffs_gpu(a_ints, b_ints, sorted_keys_t, sorted_vals_t, n_orb):
    """GPU-resident (a, b) → coeff lookup using torch.searchsorted.

    Eliminates CPU↔GPU transfer per chunk. Was the dominant cost in PT2 score
    for full_sd 52Q (~30 min just for searchsorted).
    """
    if sorted_keys_t.numel() == 0:
        return torch.zeros(a_ints.shape[0], dtype=torch.float64, device=a_ints.device)
    query_keys = (a_ints.long() << n_orb) | b_ints.long()
    idx = torch.searchsorted(sorted_keys_t, query_keys)
    idx_clipped = idx.clamp(max=sorted_keys_t.numel() - 1)
    matches = sorted_keys_t[idx_clipped] == query_keys
    coeffs = sorted_vals_t[idx_clipped]
    return torch.where(matches, coeffs, torch.zeros_like(coeffs))


def _coupling_inner(configs_chunk, sorted_keys_t, sorted_vals_t, hamiltonian, n_orb, device):
    """Inner helper: compute coupling vector for one chunk of candidates.

    All operations on GPU — no CPU↔GPU round trip.
    """
    try:
        all_connected, all_elements, batch_indices = \
            hamiltonian.get_connections_vectorized_batch(configs_chunk)
    except MemoryError:
        return None, False

    if len(all_connected) == 0:
        return torch.zeros(configs_chunk.shape[0], dtype=torch.double, device=device), True

    powers = (2 ** torch.arange(n_orb, device=device, dtype=torch.long))
    alpha_bits = all_connected[:, :n_orb].long()
    beta_bits  = all_connected[:, n_orb:].long()
    a_ints = (alpha_bits * powers).sum(dim=1)
    b_ints = (beta_bits  * powers).sum(dim=1)

    # GPU-resident searchsorted lookup (no CPU transfer)
    coeffs_t = _lookup_coeffs_gpu(a_ints, b_ints, sorted_keys_t, sorted_vals_t, n_orb)

    weighted = all_elements.double() * coeffs_t
    coupling = torch.zeros(configs_chunk.shape[0], dtype=torch.double, device=device)
    coupling.scatter_add_(0, batch_indices.long(), weighted)
    return coupling, True


def _compute_coupling_to_ground_state(new_candidates, sci_state, hamiltonian,
                                       n_orb, n_qubits, _lookup=None):
    """Compute |⟨x|H|Φ₀⟩| for each candidate config — GPU-batched.

    Strategy: pre-split candidates into chunks of size targeting ~2GB output
    each. If a chunk still raises MemoryError (rare), halve it and retry.
    This avoids the prior pathology where wasted work compounded
    exponentially across recursion levels.

    Returns array of |coupling| values (CPU numpy).
    """
    if not new_candidates:
        return np.zeros(0)

    device = hamiltonian.device

    if _lookup is None:
        _lookup = _build_coeff_lookup(sci_state, n_orb, device=device)
    sorted_keys_t, sorted_vals_t = _lookup

    # Estimate safe chunk size based on actual output volume per input,
    # NOT n_exc_max (which over-estimates by 30×).
    # Empirically: ~15k connections per config (52Q full_sd) × ~440 bytes
    #            ≈ 6.5 MB per input. Target 16 GB output per chunk.
    # Default H also has ~1.9k connections × 440B ≈ 840 KB per input.
    n_orb_attr = getattr(hamiltonian, "n_orbitals", n_orb)
    is_full_sd = bool(getattr(hamiltonian, "enumerate_zero_h", False))
    # Heuristic: n_orb² same-spin α + n_orb² same-spin β + n_orb² αβ singles
    # but for full_sd: includes ALL combinatorial pairs.
    if is_full_sd:
        # ~n_orb^2 αβ doubles ≈ n_orb² with cross-irrep all included.
        est_conn_per_input = 5 + 2 * n_orb_attr * (n_orb_attr - 1) // 2  # full αβ doubles
    else:
        # filtered: typically O(100-2000)
        est_conn_per_input = max(64, len(getattr(hamiltonian, "_single_p", [])) +
                                  len(getattr(hamiltonian, "_double_ab_p", [])) // n_orb_attr)
    bytes_per_input = max(1, est_conn_per_input * (hamiltonian.num_sites * 8 + 16))
    target_chunk_bytes = 16 * 1024**3  # 16 GB target output per chunk (matches max_output_mb)
    chunk_size = max(1, target_chunk_bytes // bytes_per_input)

    all_configs = torch.stack([c[2] for c in new_candidates]).to(device)
    N = all_configs.shape[0]

    coupling_chunks = []
    start = 0
    while start < N:
        end = min(start + chunk_size, N)
        chunk = all_configs[start:end]
        coup, ok = _coupling_inner(chunk, sorted_keys_t, sorted_vals_t, hamiltonian, n_orb, device)
        if ok:
            coupling_chunks.append(coup)
            start = end
        else:
            # Estimated chunk too big — halve and retry the same piece
            cur = end - start
            if cur <= 1:
                coupling_chunks.append(torch.zeros(1, dtype=torch.double, device=device))
                start = end
                continue
            chunk_size = max(1, chunk_size // 2)

    coupling_full = torch.cat(coupling_chunks, dim=0)
    return coupling_full.abs().cpu().numpy()


_LOSS_HISTORY = []  # filled by _update_nqs; consumed by run_*
_CURRENT_ITER = [0]  # set by v3 main loop before each _update_nqs call
_TRUST_SCHEDULE = [None]  # callable(iteration) -> trust ∈ [0,1]; None = no anneal
_USE_NQS_SCORE = [False]  # if True, replace PT2 score with NQS log_prob for selection
_USE_HCI_SCORE = [False]  # if True, replace PT2 score with HCI heat-bath: max_σ |H_σσ' c_σ|
_USE_MCSCI_SCORE = [False]  # if True, use γ-MCSCI Krylov moment score
_MCSCI_WEIGHTS = [(1.0, 0.5, 0.25)]  # (w1, w2, w3) for H¹, H², H³ moments
_MCSCI_MAX_ORDER = [2]  # 1 = CIPSI baseline, 2 = +H², 3 = +H³


def _max_h_inner(configs_chunk, sorted_keys_t, sorted_vals_t, hamiltonian, n_orb, device):
    """Inner helper for HCI max-rule on one chunk of candidates. All on GPU."""
    try:
        all_connected, all_elements, batch_indices = \
            hamiltonian.get_connections_vectorized_batch(configs_chunk)
    except MemoryError:
        return None, False

    if len(all_connected) == 0:
        return torch.zeros(configs_chunk.shape[0], dtype=torch.double, device=device), True

    powers = (2 ** torch.arange(n_orb, device=device, dtype=torch.long))
    alpha_bits = all_connected[:, :n_orb].long()
    beta_bits  = all_connected[:, n_orb:].long()
    a_ints = (alpha_bits * powers).sum(dim=1)
    b_ints = (beta_bits  * powers).sum(dim=1)

    coeffs_t = _lookup_coeffs_gpu(a_ints, b_ints, sorted_keys_t, sorted_vals_t, n_orb)

    abs_weighted = (all_elements.double() * coeffs_t).abs()
    score = torch.zeros(configs_chunk.shape[0], dtype=torch.double, device=device)
    score.scatter_reduce_(0, batch_indices.long(), abs_weighted, reduce='amax',
                          include_self=False)
    return score, True


def _compute_max_h_connection(new_candidates, sci_state, hamiltonian, n_orb, n_qubits,
                               _lookup=None):
    """HCI heat-bath score: score(σ') = max_{σ∈basis} |H_σσ' × c_σ|.

    Robust against destructive interference (PT2 sum can cancel; max can't).
    Pre-split chunking like _compute_coupling_to_ground_state.
    """
    if not new_candidates:
        return np.zeros(0)

    device = hamiltonian.device

    if _lookup is None:
        _lookup = _build_coeff_lookup(sci_state, n_orb, device=device)
    sorted_keys_t, sorted_vals_t = _lookup

    n_orb_attr = getattr(hamiltonian, "n_orbitals", n_orb)
    is_full_sd = bool(getattr(hamiltonian, "enumerate_zero_h", False))
    if is_full_sd:
        est_conn_per_input = 5 + 2 * n_orb_attr * (n_orb_attr - 1) // 2
    else:
        est_conn_per_input = max(64, len(getattr(hamiltonian, "_single_p", [])) +
                                  len(getattr(hamiltonian, "_double_ab_p", [])) // n_orb_attr)
    bytes_per_input = max(1, est_conn_per_input * (hamiltonian.num_sites * 8 + 16))
    target_chunk_bytes = 16 * 1024**3
    chunk_size = max(1, target_chunk_bytes // bytes_per_input)

    all_configs = torch.stack([c[2] for c in new_candidates]).to(device)
    N = all_configs.shape[0]

    score_chunks = []
    start = 0
    while start < N:
        end = min(start + chunk_size, N)
        chunk = all_configs[start:end]
        sc, ok = _max_h_inner(chunk, sorted_keys_t, sorted_vals_t, hamiltonian, n_orb, device)
        if ok:
            score_chunks.append(sc)
            start = end
        else:
            cur = end - start
            if cur <= 1:
                score_chunks.append(torch.zeros(1, dtype=torch.double, device=device))
                start = end
                continue
            chunk_size = max(1, chunk_size // 2)

    return torch.cat(score_chunks, dim=0).cpu().numpy()


def _update_nqs(nqs, optimizer, cumulative_bs, e0,
                sci_state, hamiltonian, cfg, device,
                n_orb, n_qubits):
    """Update NQS using eigenvector teacher + REINFORCE.

    If `_TRUST_SCHEDULE[0]` is callable, scales teacher/energy weights by
    trust(iter) and entropy weight by (2 - trust). Trust=1 reproduces
    static-weight behaviour; trust=0 disables teacher entirely.
    """
    configs = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
    n_total = len(configs)

    # Eigenvector teacher weights: joint |C[I,J]|² (exact, not marginal product).
    # Old approximation p(α_I)×p(β_J) assumes α/β independence — fails for
    # strongly correlated systems where rank(C) >> 1.
    amps = np.abs(sci_state.amplitudes) ** 2   # shape (n_alpha, n_beta)
    nonzero = np.argwhere(amps > 1e-14)
    joint_map = {
        (int(sci_state.ci_strs_a[ia]), int(sci_state.ci_strs_b[ib])): float(amps[ia, ib])
        for ia, ib in nonzero
    }

    # Vectorized bit packing (MSB-first within each spin channel)
    powers_msb = (1 << np.arange(n_orb - 1, -1, -1)).astype(np.int64)
    bs_int = np.asarray(cumulative_bs).astype(np.int64)
    a_ints = (bs_int[:, :n_orb] * powers_msb).sum(axis=1)
    b_ints = (bs_int[:, n_orb:] * powers_msb).sum(axis=1)

    teacher_weights = np.fromiter(
        (joint_map.get((int(a), int(b)), 0.0) for a, b in zip(a_ints, b_ints)),
        dtype=np.float64, count=n_total,
    )

    total_w = teacher_weights.sum()
    if total_w > 0:
        teacher_weights /= total_w
    teacher_t = torch.from_numpy(teacher_weights).float().to(device)

    # On-policy samples + PT2-augmented advantage
    # advantage(x) = (H(x,x) - E₀) - |<x|H|Ψ_CI>|² / |H(x,x) - E₀|
    # Positive = bad config (reduce prob), negative = good config (increase prob).
    # Using PT2 coupling means configs strongly connected to Ψ_CI get more negative
    # advantage even when H(x,x) is above E₀ — capturing correlation not in diagonal.
    onpol_configs_t = None
    onpol_advantage_t = None
    n_onpol = getattr(cfg, 'n_onpol_samples', 0)
    if n_onpol > 0 and sci_state is not None:
        with torch.no_grad():
            raw, _ = nqs.sample(n_onpol, temperature=1.0)
            raw_cpu = raw.long().cpu()
            a_cnt = raw_cpu[:, :n_orb].sum(dim=1)
            b_cnt = raw_cpu[:, n_orb:].sum(dim=1)
            valid = (a_cnt == hamiltonian.n_alpha) & (b_cnt == hamiltonian.n_beta)
            raw_valid = raw_cpu[valid]

        if len(raw_valid) > 0:
            onpol_unique = torch.unique(raw_valid, dim=0)
            onpol_bs = _configs_to_ibm_format(onpol_unique, n_orb, n_qubits)
            onpol_candidates = [
                (onpol_bs[i], onpol_bs[i].tobytes(), onpol_unique[i])
                for i in range(len(onpol_bs))
            ]

            with torch.no_grad():
                diag_e = hamiltonian.diagonal_elements_batch(
                    onpol_unique.float().to(device))
                diag_np = (diag_e.cpu().numpy()
                           if isinstance(diag_e, torch.Tensor)
                           else np.asarray(diag_e, dtype=np.float64))

            coupling = _compute_coupling_to_ground_state(
                onpol_candidates, sci_state, hamiltonian, n_orb, n_qubits)

            diag_adv = diag_np - e0
            # Subtract mean (REINFORCE baseline): without this, nearly all
            # on-policy configs have advantage > 0 (H(x,x) > E₀), so the
            # gradient says "reduce prob of everything" — fighting the teacher.
            # After centering: low-energy configs get negative advantage
            # (increase prob), high-energy configs get positive (decrease prob).
            _ = coupling  # kept for future local energy upgrade
            pt2_adv = diag_adv - diag_adv.mean()

            onpol_configs_t = onpol_unique.float().to(device)
            onpol_advantage_t = torch.from_numpy(
                pt2_adv.astype(np.float32)).to(device)

    max_batch = min(5000, n_total)

    for step in range(cfg.nqs_steps):
        optimizer.zero_grad()

        if n_total > max_batch:
            idx = torch.randperm(n_total)[:max_batch]
            batch_configs = configs[idx].float().to(device)
            batch_teacher = teacher_t[idx]
            batch_teacher = batch_teacher / max(batch_teacher.sum(), 1e-30)
        else:
            batch_configs = configs.float().to(device)
            batch_teacher = teacher_t

        log_probs = nqs.log_prob(batch_configs)

        loss_teacher = -(batch_teacher * log_probs).sum()

        if onpol_configs_t is not None:
            log_probs_onpol = nqs.log_prob(onpol_configs_t)
            loss_energy = (onpol_advantage_t.detach() * log_probs_onpol).mean()
        else:
            loss_energy = torch.tensor(0.0, device=device)

        loss_entropy = log_probs.mean()

        # Optional trust-annealing
        sched = _TRUST_SCHEDULE[0]
        if sched is not None:
            trust = float(sched(_CURRENT_ITER[0]))
            tw = cfg.teacher_weight * trust
            ew = cfg.energy_weight * trust
            ent_w = cfg.entropy_weight * (2.0 - trust)
        else:
            tw = cfg.teacher_weight
            ew = cfg.energy_weight
            ent_w = cfg.entropy_weight

        loss = tw * loss_teacher + ew * loss_energy + ent_w * loss_entropy

        # Record per-step loss components (consumed by run_hi_nqs_sqd_v3)
        _LOSS_HISTORY.append({
            "step": int(step),
            "L_teacher": float(loss_teacher.detach().item()),
            "L_energy":  float(loss_energy.detach().item()),
            "L_entropy": float(loss_entropy.detach().item()),
            "L_total":   float(loss.detach().item()),
        })

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
        optimizer.step()


def _ibm_row_to_int(row, n_orb, is_alpha, n_qubits):
    val = 0
    offset = 0 if is_alpha else n_orb
    for j in range(n_orb):
        if row[offset + n_orb - 1 - j]:
            val |= (1 << j)
    return val


def _configs_to_ibm_format(configs, n_orb, n_qubits):
    if torch.is_tensor(configs):
        configs_np = configs.cpu().numpy()
    else:
        configs_np = np.asarray(configs)
    bs = np.zeros((len(configs_np), n_qubits), dtype=bool)
    bs[:, :n_orb] = configs_np[:, n_orb - 1::-1].astype(bool)
    bs[:, n_orb:] = configs_np[:, 2 * n_orb - 1:n_orb - 1:-1].astype(bool)
    return bs


def _ibm_format_to_configs(bs_matrix, n_orb, n_qubits):
    bs_np = np.asarray(bs_matrix)
    configs_np = np.zeros((len(bs_np), n_qubits), dtype=np.int64)
    configs_np[:, :n_orb] = bs_np[:, n_orb - 1::-1].astype(np.int64)
    configs_np[:, n_orb:] = bs_np[:, n_qubits - 1:n_orb - 1:-1].astype(np.int64)
    return torch.from_numpy(configs_np).long()
