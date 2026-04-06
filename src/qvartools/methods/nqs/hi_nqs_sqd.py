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

from qvartools.solvers.solver import SolverResult
from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer

from qiskit_addon_sqd.fermion import solve_fermion


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
    nf_steps: int = 10
    nf_lr: float = 1e-3

    # Loss weights
    teacher_weight: float = 1.0
    energy_weight: float = 0.1
    entropy_weight: float = 0.05

    # Temperature
    initial_temperature: float = 1.0
    final_temperature: float = 0.3


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

    optimizer = torch.optim.Adam(nqs.parameters(), lr=cfg.nf_lr)

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
                h = tuple(new_bs[i].tolist())
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
            # Iter 0: no Φ₀ yet → rank by diagonal energy, take top_k
            # (will be rescored with PT2 at Iter 1)
            diag_configs = torch.stack([c[2] for c in new_candidates])
            diag_e = hamiltonian.diagonal_elements_batch(diag_configs).cpu().numpy()
            top_idx = np.argsort(diag_e)[:cfg.top_k]  # lowest energy = best

            for idx in top_idx:
                bs_row, h, _ = new_candidates[idx]
                cumulative_hashes.add(h)
                if cumulative_bs is None:
                    cumulative_bs = bs_row.reshape(1, -1)
                else:
                    cumulative_bs = np.vstack([cumulative_bs, bs_row])
            n_selected = len(top_idx)
            needs_rescore = True
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
                    all_hashes.append(tuple(cumulative_bs[i].tolist()))
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

                    for idx in top_idx:
                        bs_row, h, _ = new_candidates[idx]
                        cumulative_hashes.add(h)
                        cumulative_scores[h] = float(scores[idx])
                        cumulative_bs = np.vstack([cumulative_bs, bs_row])
                    n_selected = len(top_idx)
                else:
                    n_selected = 0

        # Seed with HF if empty
        if cumulative_bs is None:
            hf = hamiltonian.get_hf_state()
            hf_bs = _configs_to_ibm_format(hf.unsqueeze(0), n_orb, n_qubits)
            cumulative_bs = hf_bs
            cumulative_hashes.add(tuple(hf_bs[0].tolist()))

        # Evict lowest-scoring configs if over max_basis_size
        if cfg.max_basis_size > 0 and len(cumulative_bs) > cfg.max_basis_size:
            # Score all configs, keep top max_basis_size
            all_scores = np.array([
                cumulative_scores.get(tuple(cumulative_bs[i].tolist()), 0.0)
                for i in range(len(cumulative_bs))
            ])
            keep_idx = np.argsort(all_scores)[::-1][:cfg.max_basis_size]
            keep_idx.sort()
            n_evicted = len(cumulative_bs) - len(keep_idx)
            cumulative_bs = cumulative_bs[keep_idx]
            cumulative_hashes = {tuple(row.tolist()) for row in cumulative_bs}

        score_time = time.time() - score_t0

        # =====================================================
        # Step 3: SQD diagonalization (single batch, full basis)
        # =====================================================
        sqd_t0 = time.time()

        try:
            e, sci_state, occ, spin_sq = solve_fermion(
                cumulative_bs, hcore, eri, spin_sq=0,
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


def _compute_coupling_to_ground_state(new_candidates, sci_state, hamiltonian,
                                       n_orb, n_qubits):
    """Compute |⟨x|H|Φ₀⟩| for each candidate config.

    Uses the Hamiltonian's get_connections to find which basis configs
    each candidate is connected to, then weights by eigenvector coefficients.

    Returns array of |coupling| values.
    """
    # Build mapping: (alpha_int, beta_int) → eigenvector coefficient
    amps = sci_state.amplitudes  # (na, nb)
    ci_strs_a = sci_state.ci_strs_a
    ci_strs_b = sci_state.ci_strs_b

    coeff_map = {}
    for ia, a_str in enumerate(ci_strs_a):
        for ib, b_str in enumerate(ci_strs_b):
            c = amps[ia, ib]
            if abs(c) > 1e-14:
                coeff_map[(int(a_str), int(b_str))] = c

    coupling = np.zeros(len(new_candidates))

    for i, (bs_row, h, config_tensor) in enumerate(new_candidates):
        # Get H-connections from this config
        config_gpu = config_tensor.unsqueeze(0).to(hamiltonian.device)
        try:
            connected, elements, _ = hamiltonian.get_connections_vectorized_batch(config_gpu)

            # For each connected config, check if it's in the eigenvector
            total_coupling = 0.0
            for j in range(len(connected)):
                conn = connected[j]
                # Extract alpha/beta ints from connected config
                a_int = 0
                b_int = 0
                for k in range(n_orb):
                    if conn[k]:
                        a_int |= (1 << k)
                    if conn[k + n_orb]:
                        b_int |= (1 << k)

                c = coeff_map.get((a_int, b_int), 0.0)
                if c != 0.0:
                    total_coupling += float(elements[j]) * c

            coupling[i] = abs(total_coupling)
        except Exception:
            coupling[i] = 0.0

    return coupling


def _update_nqs(nqs, optimizer, cumulative_bs, e0,
                sci_state, hamiltonian, cfg, device,
                n_orb, n_qubits):
    """Update NQS using eigenvector teacher + REINFORCE."""
    configs = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
    n_total = len(configs)

    # Eigenvector teacher weights via alpha/beta marginals
    amps = np.abs(sci_state.amplitudes) ** 2
    alpha_marginal = amps.sum(axis=1)
    beta_marginal = amps.sum(axis=0)
    alpha_marginal /= max(alpha_marginal.sum(), 1e-30)
    beta_marginal /= max(beta_marginal.sum(), 1e-30)

    alpha_map = {int(s): float(v) for s, v in zip(sci_state.ci_strs_a, alpha_marginal)}
    beta_map = {int(s): float(v) for s, v in zip(sci_state.ci_strs_b, beta_marginal)}

    teacher_weights = np.zeros(n_total, dtype=np.float64)
    for i in range(n_total):
        row = cumulative_bs[i]
        a_int = _ibm_row_to_int(row, n_orb, is_alpha=True, n_qubits=n_qubits)
        b_int = _ibm_row_to_int(row, n_orb, is_alpha=False, n_qubits=n_qubits)
        teacher_weights[i] = alpha_map.get(a_int, 0.0) * beta_map.get(b_int, 0.0)

    total_w = teacher_weights.sum()
    if total_w > 0:
        teacher_weights /= total_w
    teacher_t = torch.from_numpy(teacher_weights).float().to(device)

    with torch.no_grad():
        diag_e = hamiltonian.diagonal_elements_batch(configs)
        diag_e_t = torch.tensor(np.asarray(diag_e, dtype=np.float64),
                                dtype=torch.float32, device=device)
        advantage = diag_e_t - e0

    max_batch = min(5000, n_total)

    for step in range(cfg.nf_steps):
        optimizer.zero_grad()

        if n_total > max_batch:
            idx = torch.randperm(n_total)[:max_batch]
            batch_configs = configs[idx].float().to(device)
            batch_teacher = teacher_t[idx]
            batch_teacher = batch_teacher / max(batch_teacher.sum(), 1e-30)
            batch_advantage = advantage[idx]
        else:
            batch_configs = configs.float().to(device)
            batch_teacher = teacher_t
            batch_advantage = advantage

        log_probs = nqs.log_prob(batch_configs)

        loss_teacher = -(batch_teacher * log_probs).sum()
        loss_energy = (batch_teacher * batch_advantage * log_probs).sum()
        loss_entropy = log_probs.mean()

        loss = (cfg.teacher_weight * loss_teacher
                + cfg.energy_weight * loss_energy
                + cfg.entropy_weight * loss_entropy)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
        optimizer.step()

        del batch_configs, log_probs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ibm_row_to_int(row, n_orb, is_alpha, n_qubits):
    val = 0
    offset = 0 if is_alpha else n_orb
    for j in range(n_orb):
        if row[offset + n_orb - 1 - j]:
            val |= (1 << j)
    return val


def _configs_to_ibm_format(configs, n_orb, n_qubits):
    n = len(configs)
    bs = np.zeros((n, n_qubits), dtype=bool)
    for s in range(n):
        c = configs[s]
        for j in range(n_orb):
            bs[s, n_orb - 1 - j] = bool(c[j])
            bs[s, n_qubits - 1 - j] = bool(c[j + n_orb])
    return bs


def _ibm_format_to_configs(bs_matrix, n_orb, n_qubits):
    n = len(bs_matrix)
    configs = torch.zeros(n, n_qubits, dtype=torch.long)
    for s in range(n):
        for j in range(n_orb):
            configs[s, j] = int(bs_matrix[s, n_orb - 1 - j])
            configs[s, j + n_orb] = int(bs_matrix[s, n_qubits - 1 - j])
    return configs
