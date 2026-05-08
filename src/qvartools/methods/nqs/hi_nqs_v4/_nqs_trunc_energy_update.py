"""Truncated-energy supervised update for NQS in HI+NQS+SQD.

Per Solanki, Ding, Reiher (arXiv:2602.12993): on a selected basis S,
the loss

    E^trunc_θ = Σ_{σ ∈ S} P_θ^renorm(σ) · E_loc(σ),
    P_θ^renorm(σ) = p_θ(σ) / Σ_{σ' ∈ S} p_θ(σ')

is empirically much more sample-efficient than VMC REINFORCE on
externals. Because the SQD eigenstate c satisfies the eigenvalue
equation `Σ_l H_kl c_l = e0 c_k` exactly on the basis, the truncated
local energy `E_loc(σ_k) = e0` is constant for every σ_k ∈ S — the
gradient of E^trunc therefore vanishes when p_θ ∝ c² up to a baseline
shift. The remaining gradient signal is precisely supervised KL of
p_θ against the renormalized teacher c²/Σc².

This update is the *amplitude-only* practical form: drop the REINFORCE
on couplings (which arXiv:2602.12993 reports hurts performance), keep
only the joint-c² supervised loss on the current basis, plus a small
entropy regulariser to prevent collapse during long iterations.

Compared to `nqs_vmc_update.vmc_update_nqs`, this is the same code
path with `vmc_weight = 0`; we factor it into its own function for
clarity and to avoid the cost of recomputing connections at every
gradient step.
"""
from __future__ import annotations

import numpy as np
import torch

from ._core import _ibm_format_to_configs


def trunc_energy_update_nqs(
    nqs,
    optimizer,
    cumulative_bs,
    e0,
    sci_state,
    hamiltonian,
    cfg,
    device,
    n_orb,
    n_qubits,
):
    """Truncated-energy supervised update — joint c² teacher only."""
    n_basis = len(cumulative_bs)
    if n_basis == 0:
        return

    # ----- Joint c² teacher on the current basis -----
    amps = np.asarray(sci_state.amplitudes)            # (na, nb), signed
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

    teacher_w = joint_c ** 2
    total_w = teacher_w.sum()
    if total_w > 0:
        teacher_w /= total_w
    teacher_t = torch.from_numpy(teacher_w).float().to(device)

    configs = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)

    # ----- Update loop -----
    teacher_weight = float(getattr(cfg, "teacher_weight", 1.0))
    entropy_weight = float(cfg.entropy_weight)
    max_batch = min(5000, n_basis)

    for step in range(cfg.nqs_steps):
        optimizer.zero_grad()

        # Subsample basis for the supervised loss
        if n_basis > max_batch:
            idx = torch.randperm(n_basis)[:max_batch]
            sup_cfg = configs[idx].float().to(device)
            sup_w = teacher_t[idx]
            sup_w = sup_w / max(sup_w.sum().item(), 1e-30)
        else:
            sup_cfg = configs.float().to(device)
            sup_w = teacher_t

        log_p_sup = nqs.log_prob(sup_cfg)
        sup_loss = -(sup_w * log_p_sup).sum()

        # Entropy regulariser to keep the support broad across iters
        entropy_loss = log_p_sup.mean()

        loss = teacher_weight * sup_loss + entropy_weight * entropy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
        optimizer.step()


def install_trunc_update_in_v3():
    """Monkey-patch v3/v4 _update_nqs to the truncated-energy version."""
    import src.methods.hi_nqs_sqd_v3 as _v3
    import src.methods.hi_nqs_sqd as _base
    _v3._update_nqs = trunc_energy_update_nqs
    _base._update_nqs = trunc_energy_update_nqs
    print("[trunc_energy_update] installed: joint c² supervised, no VMC reinforce",
          flush=True)
