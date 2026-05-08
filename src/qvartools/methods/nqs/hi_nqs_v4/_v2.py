"""HI+NQS+SQD v2: classical seed + per-iter classical expansion.

Changes vs v1:
  A. classical_seed (iter 0 only): basis is seeded with HF + all single+double
     excitations from HF, enumerated classically via get_connections. NQS random
     sampling then fills any remaining budget. This replaces the "random NQS
     + diagonal-energy top-k" cold start that v1 uses at iter 0.

  B. classical_expansion (every iter): after SQD, enumerate H-connections of
     the top-N amplitude dets in current basis and merge into the candidate pool
     alongside NQS samples. Keeps new important dets flowing in after NQS's
     sampling distribution collapses.

Hypothesis grounded in v1 data: every v1 NEW run (mb_100k, mb_200k, var_slowlr,
tk_50k, var_steps10, var_cold) shares iter 0 E ~= -109.105 (+110 mHa vs HCI)
because NQS random sampling + diagonal selection picks bad initial dets. OLD
incremental started at +70 mHa because n_alpha * n_beta blowup compensated.
Classical seed should bring v2 iter 0 down to single-digit mHa.
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
    HINQSSQDConfig,
    _compute_coupling_to_ground_state,
    _update_nqs,
    _configs_to_ibm_format,
    _ibm_format_to_configs,
)
from ._incremental_sqd import IncrementalSQDBackend
from ._sparse_det_backend import SparseDetSQDBackend


@dataclass
class HINQSSQDv2Config(HINQSSQDConfig):
    # A. Classical seed at iter 0
    classical_seed: bool = True
    # B. Classical expansion each iter
    classical_expansion: bool = True
    classical_expansion_top_n: int = 100       # enumerate from top-N amp dets
    classical_expansion_every_n_iters: int = 1  # do expansion every N iters (1 = always)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _enumerate_hf_and_connections(hamiltonian, n_orb, n_qubits):
    """Classical seed: HF + all singles + all doubles of HF.

    Returns IBM-format bitstring matrix (bool, n_rows x n_qubits), deduped.
    """
    hf = hamiltonian.get_hf_state().to(hamiltonian.device)
    hf_batch = hf.unsqueeze(0)
    connected, _, _ = hamiltonian.get_connections_vectorized_batch(hf_batch)
    all_configs = torch.cat([hf_batch, connected], dim=0)
    unique = torch.unique(all_configs.long().cpu(), dim=0)
    return _configs_to_ibm_format(unique, n_orb, n_qubits)


def _enumerate_top_amp_connections(sci_state, hamiltonian, n_orb, n_qubits, top_n,
                                    existing_hashes):
    """Expand singles/doubles from top-N amplitude dets in current basis.

    Returns list of (ibm_row, hash, config_tensor) for NEW candidates not
    already in existing_hashes.
    """
    amps = np.asarray(sci_state.amplitudes)
    if amps.size == 0:
        return []

    abs_flat = np.abs(amps).flatten()
    top_n = min(top_n, int((abs_flat > 1e-14).sum()))
    if top_n <= 0:
        return []

    # Top indices by amplitude
    top_flat = np.argpartition(-abs_flat, top_n - 1)[:top_n]
    n_b = amps.shape[1]
    top_ia = top_flat // n_b
    top_ib = top_flat % n_b

    ci_strs_a = np.asarray(sci_state.ci_strs_a)
    ci_strs_b = np.asarray(sci_state.ci_strs_b)

    # (alpha_int, beta_int) pairs -> NQS configs (bit k = orbital k occupancy)
    N = len(top_ia)
    configs_np = np.zeros((N, n_qubits), dtype=np.int64)
    for i in range(N):
        a = int(ci_strs_a[top_ia[i]])
        b = int(ci_strs_b[top_ib[i]])
        for k in range(n_orb):
            configs_np[i, k] = (a >> k) & 1
            configs_np[i, n_orb + k] = (b >> k) & 1

    configs_t = torch.from_numpy(configs_np).long().to(hamiltonian.device)

    # Enumerate connections of these top-amp dets
    try:
        connected, _, _ = hamiltonian.get_connections_vectorized_batch(configs_t)
    except (MemoryError, RuntimeError) as ex:
        # fallback: chunk
        chunks = []
        chunk_size = max(1, N // 4)
        for start in range(0, N, chunk_size):
            c, _, _ = hamiltonian.get_connections_vectorized_batch(
                configs_t[start:start + chunk_size]
            )
            chunks.append(c.cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        connected = torch.cat(chunks, dim=0) if chunks else torch.zeros(
            (0, n_qubits), dtype=torch.long
        )

    if len(connected) == 0:
        return []

    # GPU-resident dedup: pack each row to int64 hash, sort+diff for uniqueness.
    # Faster than CPU torch.unique(dim=0) on millions of 52-element rows.
    # Move to CPU at end (caller convention: connected_unique is CPU tensor).
    if torch.cuda.is_available() and connected.device.type == "cpu":
        connected = connected.to(hamiltonian.device)
    connected_long = connected.long()
    pack_powers = (1 << torch.arange(n_qubits, device=connected_long.device, dtype=torch.long))
    packed_keys = (connected_long * pack_powers).sum(dim=1)         # (N,) int64 hash
    sorted_keys, sort_idx = torch.sort(packed_keys)
    is_unique = torch.cat([
        torch.ones(1, dtype=torch.bool, device=connected_long.device),
        sorted_keys[1:] != sorted_keys[:-1],
    ])
    unique_pos = sort_idx[is_unique]
    connected_unique = connected_long[unique_pos].cpu()  # back to CPU for caller
    ibm_bs = _configs_to_ibm_format(connected_unique, n_orb, n_qubits)

    # Filter to NEW only (set lookup unavoidable, but only over deduplicated set)
    out = []
    for i in range(len(ibm_bs)):
        h = ibm_bs[i].tobytes()
        if h not in existing_hashes:
            out.append((ibm_bs[i], h, connected_unique[i]))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_hi_nqs_sqd_v2(hamiltonian, mol_info,
                      config: Optional[HINQSSQDv2Config] = None) -> SolverResult:
    """HI+NQS+SQD v2. See module docstring for v1 -> v2 diff."""
    t0 = time.time()
    cfg = config or HINQSSQDv2Config()

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
    print(f"    HI+NQS+SQD v2 (GPU={device}): arch={embed}/{heads}/{layers}, "
          f"params={n_params:,}, samples={cfg.n_samples}, top_k={cfg.top_k}, "
          f"classical_seed={cfg.classical_seed}, "
          f"classical_expansion={cfg.classical_expansion}", flush=True)

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

    # Classical seed: pre-populate with HF + singles + doubles
    if cfg.classical_seed:
        seed_t0 = time.time()
        seed_bs = _enumerate_hf_and_connections(hamiltonian, n_orb, n_qubits)
        cumulative_bs = seed_bs
        for i in range(len(seed_bs)):
            cumulative_hashes.add(seed_bs[i].tobytes())
        print(f"    [v2 classical_seed] HF + singles + doubles: {len(seed_bs):,} dets "
              f"(enum time {time.time()-seed_t0:.2f}s)", flush=True)

    # SQD backend selection
    sqd_backend = None
    if cfg.use_sparse_det_solver:
        sqd_backend = SparseDetSQDBackend(
            hcore=hcore, eri=eri, n_alpha=n_alpha, n_beta=n_beta,
        )
    elif cfg.use_incremental_sqd:
        sqd_backend = IncrementalSQDBackend(
            hcore=hcore, eri=eri, n_alpha=n_alpha, n_beta=n_beta, spin_sq=0,
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

        # =====================================================
        # Step 1.5 (v2): classical expansion — enumerate connections of
        # top-N amplitude dets. Skip at iter 0 (no sci_state yet) and if disabled.
        # =====================================================
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
            # Dedup against NQS-sample new_candidates too
            sample_hashes = {c[1] for c in new_candidates}
            for row, h, cfg_t in expanded:
                if h not in sample_hashes:
                    new_candidates.append((row, h, cfg_t))
                    sample_hashes.add(h)
            expand_added = len(expanded)
        expand_time = time.time() - expand_t0

        # =====================================================
        # Step 2: PT2 scoring + top-k selection
        # =====================================================
        score_t0 = time.time()
        n_selected = 0
        n_evicted = 0

        if not new_candidates and not needs_rescore:
            n_selected = 0
        elif prev_sci_state is None or current_e0 is None:
            # Iter 0 path: if classical_seed pre-populated, rank NEW NQS candidates
            # by diagonal energy and append top cold_start_k. If monotonic, keep
            # fewer to avoid noise bloat.
            diag_configs = torch.stack([c[2] for c in new_candidates])
            diag_e = hamiltonian.diagonal_elements_batch(diag_configs).cpu().numpy()

            if cfg.monotonic_basis:
                _csk = (cfg.cold_start_k if cfg.cold_start_k is not None
                        else max(1, cfg.top_k // 4))
            else:
                _csk = (cfg.cold_start_k if cfg.cold_start_k is not None
                        else cfg.top_k)
            # v2: if classical_seed already took some of the budget, fill remainder
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
        # Step 3: SQD diagonalization
        # =====================================================
        sqd_t0 = time.time()
        try:
            if sqd_backend is not None:
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
        except Exception as ex:
            print(f"    Iter {iteration:>3d}: SQD failed: {ex}", flush=True)
            continue

        sqd_time = time.time() - sqd_t0

        if e0 < best_energy:
            best_energy = e0

        energy_history.append(e0)
        basis_size_history.append(len(cumulative_bs))

        # =====================================================
        # Step 4: Update NQS
        # =====================================================
        update_t0 = time.time()
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

    wall_time = time.time() - t0

    return SolverResult(
        energy=best_energy if best_energy < float("inf") else None,
        diag_dim=len(cumulative_bs) if cumulative_bs is not None else 0,
        wall_time=wall_time,
        method="HI+NQS+SQD-v2",
        converged=converged,
        metadata={
            "iterations": iteration + 1 if "iteration" in dir() else 0,
            "energy_history": energy_history,
            "basis_size_history": basis_size_history,
            "device": device,
            "v2_classical_seed": cfg.classical_seed,
            "v2_classical_expansion": cfg.classical_expansion,
        },
    )
