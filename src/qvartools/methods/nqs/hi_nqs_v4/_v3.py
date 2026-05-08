"""HI+NQS+SQD v3: v2_expand + final PT2 correction + larger expansion pool.

Changes vs v2:
  C. Final Epstein-Nesbet PT2 correction after variational convergence:
       E_PT2 = sum_x |<x|H|Psi_0>|^2 / (E_0 - H_xx)
     where x runs over external dets connected to top-N amplitude basis dets.
     Applied once at end; no cost during iteration loop. Standard CIPSI / HCI
     technique; typical gain 5-10x reduction in error.

  D. classical_expansion_top_n default 1000 (was 100). Bigger pool -> more
     diverse candidates -> basis keeps growing past v2's ~140k plateau.

  E. Looser convergence defaults: threshold=1e-9, window=5, max_iter=50.
     So we don't stop at iter 8 when basis is still growing usefully.

  Also drops classical_seed default to False (v2 data shows it hurts at large
  budgets due to mode collapse).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from qiskit_addon_sqd.fermion import solve_fermion, bitstring_matrix_to_ci_strs
from pyscf.fci.selected_ci import SCIvector

from ._solver_result import SolverResult
from ._transformer import AutoregressiveTransformer
from ._core import (
    _compute_coupling_to_ground_state,
    _update_nqs,
    _configs_to_ibm_format,
    _ibm_format_to_configs,
)
from ._v2 import (
    HINQSSQDv2Config,
    _enumerate_hf_and_connections,
    _enumerate_top_amp_connections,
)
from ._incremental_sqd import IncrementalSQDBackend
from ._sparse_det_backend import SparseDetSQDBackend


@dataclass
class HINQSSQDv3Config(HINQSSQDv2Config):
    # v3 defaults — looser convergence, bigger expansion pool, PT2 correction
    convergence_threshold: float = 1e-9
    convergence_window: int = 5
    max_iterations: int = 50
    classical_seed: bool = False                 # off by default (hurts at large budget)
    classical_expansion: bool = True
    classical_expansion_top_n: int = 1000        # 10x v2
    classical_expansion_every_n_iters: int = 1
    # New in v3
    final_pt2_correction: bool = True
    pt2_top_n: int = 5000                        # expand from top-N amp basis dets for PT2
    pt2_chunk: int = 200                         # process chunk of top dets at once
    # NEW: eviction policy. PT2 sum-rule cancels signs for spin-coupled (α=1,β=1)
    # doubles → important π/π* dets get evicted. Using |c|² from prev sci_state
    # for eviction preserves high-amplitude dets regardless of PT2 sign cancellation.
    c2_eviction: bool = False


# -----------------------------------------------------------------------------
def _final_pt2_correction(cumulative_bs, sci_state, hamiltonian,
                           n_orb, n_qubits, e0, pt2_top_n, chunk):
    """Epstein-Nesbet PT2 correction over externals connected to top-N basis dets.

    Returns (E_PT2, n_externals_processed).
    """
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
    seen_external = set(basis_hashes)

    e_pt2 = 0.0
    n_ext_total = 0

    for start in range(0, top_n, chunk):
        end = min(start + chunk, top_n)
        batch = configs_t[start:end].to(hamiltonian.device)

        try:
            connected, _, _ = hamiltonian.get_connections_vectorized_batch(batch)
        except (MemoryError, RuntimeError):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if len(connected) == 0:
            continue

        conn_unique = torch.unique(connected.long().cpu(), dim=0)
        conn_ibm = _configs_to_ibm_format(conn_unique, n_orb, n_qubits)

        # Keep only NEW external dets (dedup across all chunks)
        ext_list = []
        for i in range(len(conn_ibm)):
            h = conn_ibm[i].tobytes()
            if h not in seen_external:
                seen_external.add(h)
                ext_list.append((conn_ibm[i], h, conn_unique[i]))

        if not ext_list:
            continue

        # Compute coupling <x|H|Psi_0> for each external
        coupling = _compute_coupling_to_ground_state(
            ext_list, sci_state, hamiltonian, n_orb, n_qubits
        )

        # H_xx for each external
        ext_configs = torch.stack([c[2] for c in ext_list])
        h_diag = hamiltonian.diagonal_elements_batch(ext_configs).cpu().numpy()

        denom = e0 - h_diag  # expected < 0 for external (higher energy)
        safe = np.abs(denom) > 1e-12
        chunk_pt2 = float(np.sum((coupling[safe] ** 2) / denom[safe]))

        e_pt2 += chunk_pt2
        n_ext_total += int(safe.sum())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return e_pt2, n_ext_total


# -----------------------------------------------------------------------------
def run_hi_nqs_sqd_v3(hamiltonian, mol_info,
                      config: Optional[HINQSSQDv3Config] = None) -> SolverResult:
    """HI+NQS+SQD v3. See module docstring for v2 -> v3 diff."""
    t0 = time.time()
    cfg = config or HINQSSQDv3Config()

    # Reset loss history for this run (cleared again at end before drain)
    from ._core import _LOSS_HISTORY
    _LOSS_HISTORY.clear()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    n_qubits = 2 * n_orb

    integrals = hamiltonian.integrals
    hcore = np.asarray(integrals.h1e, dtype=np.float64)
    eri = np.asarray(integrals.h2e, dtype=np.float64)
    nuclear_repulsion = float(integrals.nuclear_repulsion)

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
          f"params={n_params:,}, samples={cfg.n_samples}, top_k={cfg.top_k}, "
          f"classical_seed={cfg.classical_seed}, "
          f"expand_top_n={cfg.classical_expansion_top_n}, "
          f"final_pt2={cfg.final_pt2_correction}", flush=True)

    energy_history = []
    basis_size_history = []
    prev_energy = float("inf")
    best_energy = float("inf")
    converged = False
    converge_count = 0

    cumulative_bs = None
    cumulative_hashes = set()
    cumulative_scores = {}

    prev_sci_state = None
    current_e0 = None
    needs_rescore = False

    if cfg.classical_seed:
        seed_t0 = time.time()
        seed_bs = _enumerate_hf_and_connections(hamiltonian, n_orb, n_qubits)
        cumulative_bs = seed_bs
        for i in range(len(seed_bs)):
            cumulative_hashes.add(seed_bs[i].tobytes())
        print(f"    [v3 classical_seed] HF+S+D: {len(seed_bs):,} dets "
              f"(t={time.time()-seed_t0:.2f}s)", flush=True)

    sqd_backend = None
    if cfg.use_sparse_det_solver:
        sqd_backend = SparseDetSQDBackend(
            hcore=hcore, eri=eri, n_alpha=n_alpha, n_beta=n_beta,
        )
    elif cfg.use_incremental_sqd:
        sqd_backend = IncrementalSQDBackend(
            hcore=hcore, eri=eri, n_alpha=n_alpha, n_beta=n_beta, spin_sq=0,
        )

    last_sci_state = None  # track for PT2

    for iteration in range(cfg.max_iterations):
        iter_t0 = time.time()
        progress = iteration / max(cfg.max_iterations - 1, 1)
        temperature = (cfg.initial_temperature
                       + progress * (cfg.final_temperature - cfg.initial_temperature))

        # Step 1: NQS sampling
        sample_t0 = time.time()
        with torch.no_grad():
            configs_gpu, _ = nqs.sample(cfg.n_samples, temperature=temperature)
            configs_cpu = configs_gpu.long().cpu()
            alpha_counts = configs_cpu[:, :n_orb].sum(dim=1)
            beta_counts = configs_cpu[:, n_orb:].sum(dim=1)
            valid = (alpha_counts == n_alpha) & (beta_counts == n_beta)
            new_configs = configs_cpu[valid]

        new_candidates = []
        if len(new_configs) > 0:
            new_unique = torch.unique(new_configs, dim=0)
            new_bs = _configs_to_ibm_format(new_unique, n_orb, n_qubits)
            for i in range(len(new_bs)):
                h = new_bs[i].tobytes()
                if h not in cumulative_hashes:
                    new_candidates.append((new_bs[i], h, new_unique[i]))
        sample_time = time.time() - sample_t0

        # Step 1.5: classical expansion
        expand_t0 = time.time()
        expand_added = 0
        if (cfg.classical_expansion
                and prev_sci_state is not None
                and iteration % cfg.classical_expansion_every_n_iters == 0):
            expanded = _enumerate_top_amp_connections(
                prev_sci_state, hamiltonian,
                n_orb, n_qubits,
                top_n=cfg.classical_expansion_top_n,
                existing_hashes=cumulative_hashes,
            )
            sample_hashes = {c[1] for c in new_candidates}
            for row, h, cfg_t in expanded:
                if h not in sample_hashes:
                    new_candidates.append((row, h, cfg_t))
                    sample_hashes.add(h)
            expand_added = len(expanded)
        expand_time = time.time() - expand_t0

        # Step 2: PT2 scoring + top-k selection
        score_t0 = time.time()
        n_selected = 0
        n_evicted = 0

        if not new_candidates and not needs_rescore:
            n_selected = 0
        elif prev_sci_state is None or current_e0 is None:
            diag_configs = torch.stack([c[2] for c in new_candidates])
            diag_e = hamiltonian.diagonal_elements_batch(diag_configs).cpu().numpy()

            if cfg.monotonic_basis:
                _csk = (cfg.cold_start_k if cfg.cold_start_k is not None
                        else max(1, cfg.top_k // 4))
            else:
                _csk = (cfg.cold_start_k if cfg.cold_start_k is not None
                        else cfg.top_k)
            if cfg.classical_seed and cumulative_bs is not None:
                already = len(cumulative_bs)
                remaining = max(0, _csk - already)
                _csk = remaining
            keep_n = min(_csk, len(diag_e))

            if keep_n > 0:
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
            if needs_rescore and cumulative_bs is not None:
                all_bs = []
                all_hashes = []
                all_configs = []
                existing_configs = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
                for i in range(len(cumulative_bs)):
                    all_bs.append(cumulative_bs[i])
                    all_hashes.append(cumulative_bs[i].tobytes())
                    all_configs.append(existing_configs[i])
                for bs_row, h, config_tensor in new_candidates:
                    all_bs.append(bs_row)
                    all_hashes.append(h)
                    all_configs.append(config_tensor)

                all_as_candidates = [
                    (all_bs[i], all_hashes[i], all_configs[i])
                    for i in range(len(all_bs))
                ]
                diag_configs = torch.stack(all_configs)
                diag_e = hamiltonian.diagonal_elements_batch(diag_configs).cpu().numpy()

                from ._core import _USE_NQS_SCORE as _UNS
                from ._core import _USE_HCI_SCORE as _UHS
                from ._core import _USE_MCSCI_SCORE as _UMS
                from ._core import _MCSCI_WEIGHTS, _MCSCI_MAX_ORDER
                from ._core import _compute_max_h_connection
                from ._nqs_mcsci_score import compute_mcsci_score
                if _UMS[0]:
                    scores = compute_mcsci_score(
                        all_as_candidates, prev_sci_state, hamiltonian,
                        n_orb, n_qubits,
                        weights=_MCSCI_WEIGHTS[0], max_order=_MCSCI_MAX_ORDER[0])
                elif _UNS[0]:
                    with torch.no_grad():
                        chunks = []
                        bs = 4096
                        for s in range(0, diag_configs.shape[0], bs):
                            cfg_chunk = diag_configs[s:s+bs].float().to(device)
                            chunks.append(nqs.log_prob(cfg_chunk).detach().cpu())
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        log_p = torch.cat(chunks, dim=0)
                    scores = log_p.numpy()
                elif _UHS[0]:
                    # HCI heat-bath: max_σ |H_σσ' × c_σ|
                    scores = _compute_max_h_connection(
                        all_as_candidates, prev_sci_state, hamiltonian, n_orb, n_qubits)
                else:
                    coupling = _compute_coupling_to_ground_state(
                        all_as_candidates, prev_sci_state, hamiltonian, n_orb, n_qubits
                    )
                    scores = np.zeros(len(all_as_candidates))
                    for i in range(len(all_as_candidates)):
                        denom = abs(current_e0 - diag_e[i])
                        if denom < 1e-12:
                            denom = 1e-12
                        scores[i] = coupling[i] ** 2 / denom

                keep_n = min(cfg.top_k, len(scores))
                top_idx = np.argsort(scores)[::-1][:keep_n]

                cumulative_bs = np.stack([all_bs[i] for i in top_idx])
                cumulative_hashes = {all_hashes[i] for i in top_idx}
                cumulative_scores = {all_hashes[i]: float(scores[i]) for i in top_idx}

                n_selected = len(top_idx) - len(existing_configs)
                n_evicted = len(existing_configs) + len(new_candidates) - len(top_idx)
                needs_rescore = False
            else:
                if new_candidates:
                    diag_configs = torch.stack([c[2] for c in new_candidates])
                    diag_e = hamiltonian.diagonal_elements_batch(diag_configs).cpu().numpy()

                    from ._core import _USE_NQS_SCORE as _UNS2
                    from ._core import _USE_HCI_SCORE as _UHS2
                    from ._core import _USE_MCSCI_SCORE as _UMS2
                    from ._core import _MCSCI_WEIGHTS as _MW2, _MCSCI_MAX_ORDER as _MO2
                    from ._core import _compute_max_h_connection as _cmh2
                    from ._nqs_mcsci_score import compute_mcsci_score as _mcsci2
                    if _UMS2[0]:
                        scores = _mcsci2(new_candidates, prev_sci_state,
                                         hamiltonian, n_orb, n_qubits,
                                         weights=_MW2[0], max_order=_MO2[0])
                    elif _UNS2[0]:
                        with torch.no_grad():
                            chunks = []
                            bs = 4096
                            for s in range(0, diag_configs.shape[0], bs):
                                cfg_chunk = diag_configs[s:s+bs].float().to(device)
                                chunks.append(nqs.log_prob(cfg_chunk).detach().cpu())
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            log_p = torch.cat(chunks, dim=0)
                        scores = log_p.numpy()
                    elif _UHS2[0]:
                        scores = _cmh2(new_candidates, prev_sci_state,
                                       hamiltonian, n_orb, n_qubits)
                    else:
                        coupling = _compute_coupling_to_ground_state(
                            new_candidates, prev_sci_state, hamiltonian, n_orb, n_qubits
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

        if cumulative_bs is None:
            hf = hamiltonian.get_hf_state()
            hf_bs = _configs_to_ibm_format(hf.unsqueeze(0), n_orb, n_qubits)
            cumulative_bs = hf_bs
            cumulative_hashes.add(hf_bs[0].tobytes())

        if (not cfg.monotonic_basis
                and cfg.max_basis_size > 0
                and len(cumulative_bs) > cfg.max_basis_size):
            # Eviction policy: PT2 (default) or |c|² from prev sci_state (c2_eviction).
            use_c2 = (getattr(cfg, 'c2_eviction', False)
                      and prev_sci_state is not None)
            if use_c2:
                # Use |c|² from prev sci_state for eviction; falls back to PT2
                # for newly added dets not yet in prev_sci_state.
                amps2 = np.abs(prev_sci_state.amplitudes) ** 2
                _powers = (1 << np.arange(n_orb - 1, -1, -1)).astype(np.int64)
                _bs_int = np.asarray(cumulative_bs).astype(np.int64)
                _a_ints = (_bs_int[:, :n_orb] * _powers).sum(axis=1)
                _b_ints = (_bs_int[:, n_orb:] * _powers).sum(axis=1)
                _a_lookup = {int(s): i for i, s in enumerate(prev_sci_state.ci_strs_a)}
                _b_lookup = {int(s): i for i, s in enumerate(prev_sci_state.ci_strs_b)}
                # Normalize PT2 fallback to comparable scale with |c|²:
                # use the median |c|² as floor for missing entries.
                pt2_scores_raw = np.array([
                    cumulative_scores.get(cumulative_bs[i].tobytes(), 0.0)
                    for i in range(len(cumulative_bs))
                ])
                # Build c² scores
                all_scores = np.zeros(len(cumulative_bs))
                _have_c2 = np.zeros(len(cumulative_bs), dtype=bool)
                for _i in range(len(cumulative_bs)):
                    _ai = _a_lookup.get(int(_a_ints[_i]))
                    _bi = _b_lookup.get(int(_b_ints[_i]))
                    if _ai is not None and _bi is not None:
                        all_scores[_i] = float(amps2[_ai, _bi])
                        _have_c2[_i] = True
                # For dets without c² (newly added): use PT2 score scaled to
                # the same magnitude as |c|² range (so they compete fairly).
                if (~_have_c2).any() and _have_c2.any():
                    c2_scale = float(all_scores[_have_c2].max() + 1e-30)
                    pt2_scale = float(pt2_scores_raw[~_have_c2].max() + 1e-30)
                    all_scores[~_have_c2] = pt2_scores_raw[~_have_c2] * (c2_scale / pt2_scale)
                elif (~_have_c2).any():
                    all_scores[~_have_c2] = pt2_scores_raw[~_have_c2]
            else:
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

        # Step 3: SQD
        sqd_t0 = time.time()
        try:
            if sqd_backend is not None:
                # If warm_start disabled, clear backend's prev-amps cache so
                # Davidson starts fresh each iter (avoids lock-in to wrong
                # eigenvalue when basis growth weakens H-coupling to prev vector).
                if not cfg.warm_start and hasattr(sqd_backend, '_prev_amps'):
                    sqd_backend._prev_amps = {}
                e, sci_state = sqd_backend.solve(cumulative_bs)
                e0 = e + nuclear_repulsion
            else:
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
            last_sci_state = sci_state
        except Exception as ex:
            print(f"    Iter {iteration:>3d}: SQD failed: {ex}", flush=True)
            continue

        sqd_time = time.time() - sqd_t0

        if e0 < best_energy:
            best_energy = e0

        energy_history.append(e0)
        basis_size_history.append(len(cumulative_bs))

        # Step 4: update NQS
        update_t0 = time.time()
        from ._core import _CURRENT_ITER as _CITER
        _CITER[0] = iteration
        _update_nqs(
            nqs, optimizer, cumulative_bs, e0,
            sci_state, hamiltonian, cfg, device, n_orb, n_qubits,
        )
        update_time = time.time() - update_t0

        delta_e = abs(e0 - prev_energy)
        prev_energy = e0
        iter_time = time.time() - iter_t0

        if delta_e < cfg.convergence_threshold and iteration > 0:
            converge_count += 1
        else:
            converge_count = 0

        print(f"    Iter {iteration:>3d}: E={e0:.10f}, "
              f"basis={len(cumulative_bs):>6d}(+{n_selected}, -{n_evicted}), "
              f"expand=+{expand_added}, "
              f"ΔE={delta_e:.2e}, "
              f"t={iter_time:.1f}s [samp={sample_time:.1f} exp={expand_time:.1f} "
              f"pt2={score_time:.1f} sqd={sqd_time:.1f} upd={update_time:.1f}]",
              flush=True)

        if converge_count >= cfg.convergence_window:
            converged = True
            break

    var_wall = time.time() - t0

    # -------------------------------------------------------------------------
    # Final PT2 correction
    # -------------------------------------------------------------------------
    e_pt2 = 0.0
    n_ext = 0
    pt2_wall = 0.0
    e_total = best_energy

    if cfg.final_pt2_correction and last_sci_state is not None and best_energy < float("inf"):
        pt2_t0 = time.time()
        print(f"    [v3 PT2] expanding from top {cfg.pt2_top_n} amp basis dets ...",
              flush=True)
        e_pt2, n_ext = _final_pt2_correction(
            cumulative_bs, last_sci_state, hamiltonian,
            n_orb, n_qubits, best_energy,
            pt2_top_n=cfg.pt2_top_n,
            chunk=cfg.pt2_chunk,
        )
        pt2_wall = time.time() - pt2_t0
        e_total = best_energy + e_pt2
        print(f"    [v3 PT2] E_var     = {best_energy:.10f}", flush=True)
        print(f"    [v3 PT2] E_PT2     = {e_pt2:+.10f}  "
              f"({n_ext:,} externals, t={pt2_wall:.1f}s)", flush=True)
        print(f"    [v3 PT2] E_total   = {e_total:.10f}", flush=True)

    wall_time = time.time() - t0

    # Drain loss history captured by _update_nqs
    from ._core import _LOSS_HISTORY
    loss_history = list(_LOSS_HISTORY)
    _LOSS_HISTORY.clear()

    return SolverResult(
        energy=e_total if e_total < float("inf") else None,
        diag_dim=len(cumulative_bs) if cumulative_bs is not None else 0,
        wall_time=wall_time,
        method="HI+NQS+SQD-v3",
        converged=converged,
        metadata={
            "iterations": iteration + 1 if "iteration" in dir() else 0,
            "energy_history": energy_history,
            "loss_history": loss_history,
            "basis_size_history": basis_size_history,
            "final_basis": cumulative_bs.copy() if cumulative_bs is not None else None,
            "device": device,
            "v3_classical_seed": cfg.classical_seed,
            "v3_classical_expansion": cfg.classical_expansion,
            "v3_expand_top_n": cfg.classical_expansion_top_n,
            "v3_final_pt2_correction": cfg.final_pt2_correction,
            "v3_pt2_top_n": cfg.pt2_top_n,
            "e_var": best_energy,
            "e_pt2": e_pt2,
            "e_total": e_total,
            "n_pt2_externals": n_ext,
            "var_wall_s": var_wall,
            "pt2_wall_s": pt2_wall,
        },
    )
