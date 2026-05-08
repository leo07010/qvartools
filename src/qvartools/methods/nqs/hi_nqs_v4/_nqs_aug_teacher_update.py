"""Augmented-teacher NQS update.

Trains NQS on the union of:
  (a) basis dets with weight = c²(σ)            ← standard joint-c² teacher
  (b) Hamiltonian extension Ĥ·basis (excluding basis itself) with weight
      = α · |Σ_b H_σ'b · c_b|²                  ← coupling-amplitude squared

Goal: explicitly teach NQS that high-excitation dets (T, Q, ...) reachable
via S+D from basis are important, even though they're not currently in
the diagonalization subspace.

Hypothesis: for stretched N₂ where ground state has c²(Q) ~ 0.16, the
H-extension surfaces those Q dets to NQS during training, enabling NQS
to sample them in next iter (instead of staying stuck in S+D-closed
subspace).
"""
from __future__ import annotations

import numpy as np
import torch

from ._core import _ibm_format_to_configs


def aug_teacher_update_nqs(
    nqs, optimizer, cumulative_bs, e0,
    sci_state, hamiltonian, cfg, device, n_orb, n_qubits,
):
    n_basis = len(cumulative_bs)
    if n_basis == 0:
        return

    # ----- Joint c² teacher on basis -----
    amps = np.asarray(sci_state.amplitudes)
    ci_strs_a = np.asarray(sci_state.ci_strs_a)
    ci_strs_b = np.asarray(sci_state.ci_strs_b)

    powers_msb = (1 << np.arange(n_orb - 1, -1, -1)).astype(np.int64)
    bs_int = np.asarray(cumulative_bs).astype(np.int64)
    a_ints = (bs_int[:, :n_orb] * powers_msb).sum(axis=1)
    b_ints = (bs_int[:, n_orb:] * powers_msb).sum(axis=1)

    a_to_ia = {int(s): i for i, s in enumerate(ci_strs_a)}
    b_to_ib = {int(s): i for i, s in enumerate(ci_strs_b)}

    joint_c = np.zeros(n_basis, dtype=np.float64)
    for k in range(n_basis):
        ia = a_to_ia.get(int(a_ints[k]), -1)
        ib = b_to_ib.get(int(b_ints[k]), -1)
        if ia >= 0 and ib >= 0:
            joint_c[k] = amps[ia, ib]

    basis_configs = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
    basis_w = (joint_c ** 2).astype(np.float64)

    # ----- H-extension via S+D from basis -----
    powers_lsb = (1 << np.arange(n_orb)).astype(np.int64)
    a_lsb = (bs_int[:, :n_orb] * powers_lsb).sum(axis=1)
    b_lsb = (bs_int[:, n_orb:] * powers_lsb).sum(axis=1)
    basis_keys = set(zip(a_lsb.tolist(), b_lsb.tolist()))

    # Compute connections in batches
    with torch.no_grad():
        try:
            connected, elements, src_idx = (
                hamiltonian.get_connections_vectorized_batch(
                    basis_configs.float().to(device)
                )
            )
        except (RuntimeError, MemoryError):
            connected = torch.zeros((0, 2*n_orb), dtype=torch.long, device=device)
            elements = torch.zeros(0, device=device)
            src_idx = torch.zeros(0, dtype=torch.long, device=device)

    n_ext_total = connected.shape[0] if connected.dim() > 1 else 0

    # Aggregate coupling per unique extension det:
    #     coupling(σ') = Σ_b H_σ'b · c_b  for b in basis, σ' connected to b
    if n_ext_total > 0:
        powers_lsb_t = torch.tensor(powers_lsb, device=device, dtype=torch.long)
        ext_a = (connected[:, :n_orb].long() * powers_lsb_t).sum(dim=1)
        ext_b = (connected[:, n_orb:].long() * powers_lsb_t).sum(dim=1)
        ext_a_np = ext_a.cpu().numpy()
        ext_b_np = ext_b.cpu().numpy()
        elem_np = elements.detach().cpu().numpy().astype(np.float64)
        src_np = src_idx.detach().cpu().numpy()

        c_src = joint_c[src_np]            # c_basis[source]  (signed)
        contrib = elem_np * c_src           # H_σ'b · c_b per pair

        # Group by extension det (a_lsb, b_lsb), summing contrib
        # and filter out σ' already in basis
        ext_coupling = {}
        for k in range(n_ext_total):
            key = (int(ext_a_np[k]), int(ext_b_np[k]))
            if key in basis_keys:
                continue
            if key not in ext_coupling:
                ext_coupling[key] = [0.0, connected[k].long().cpu().numpy()]
            ext_coupling[key][0] += contrib[k]

        ext_keys = list(ext_coupling.keys())
        if ext_keys:
            ext_couplings = np.array([ext_coupling[k][0] for k in ext_keys],
                                     dtype=np.float64)
            ext_configs_np = np.stack([ext_coupling[k][1] for k in ext_keys])
            ext_w = ext_couplings ** 2
            ext_configs = torch.from_numpy(ext_configs_np).long().to(device)
        else:
            ext_w = np.zeros(0, dtype=np.float64)
            ext_configs = torch.zeros((0, 2*n_orb), dtype=torch.long, device=device)
    else:
        ext_w = np.zeros(0, dtype=np.float64)
        ext_configs = torch.zeros((0, 2*n_orb), dtype=torch.long, device=device)

    # Combine teacher: basis + extension. Re-weight so extension contributes α
    # of total mass (default α = 0.5 means "half the supervision is on Q-style
    # extension dets")
    aug_alpha = float(getattr(cfg, "aug_extension_weight", 0.5))
    sum_basis = basis_w.sum() + 1e-30
    sum_ext   = ext_w.sum() + 1e-30

    # Normalize within each group, then mix
    basis_w_norm = basis_w / sum_basis * (1.0 - aug_alpha)
    ext_w_norm   = ext_w   / sum_ext   * aug_alpha

    teacher_w = np.concatenate([basis_w_norm, ext_w_norm])
    all_configs = torch.cat([basis_configs.long().to(device), ext_configs], dim=0)
    teacher_t = torch.from_numpy(teacher_w).float().to(device)

    n_total = len(teacher_t)
    print(f"[aug_teacher] basis={n_basis}, extension={len(ext_w)}, "
          f"total_teacher={n_total}, aug_α={aug_alpha}", flush=True)

    # ----- Update loop -----
    teacher_weight = float(cfg.teacher_weight)
    entropy_weight = float(cfg.entropy_weight)
    max_batch = min(5000, n_total)

    for step in range(cfg.nqs_steps):
        optimizer.zero_grad()

        if n_total > max_batch:
            idx = torch.randperm(n_total)[:max_batch]
            sup_cfg = all_configs[idx].float()
            sup_w = teacher_t[idx]
            sup_w = sup_w / max(sup_w.sum().item(), 1e-30)
        else:
            sup_cfg = all_configs.float()
            sup_w = teacher_t

        log_p_sup = nqs.log_prob(sup_cfg)
        sup_loss = -(sup_w * log_p_sup).sum()
        entropy_loss = log_p_sup.mean()

        loss = teacher_weight * sup_loss + entropy_weight * entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
        optimizer.step()


def install_aug_teacher_update_in_v3():
    from . import _v3
    from . import _core as _base
    _v3._update_nqs = aug_teacher_update_nqs
    _base._update_nqs = aug_teacher_update_nqs
    print("[aug_teacher] installed: supervised on basis + Ĥ·basis extension",
          flush=True)
