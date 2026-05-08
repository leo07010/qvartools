"""SR-based NQS update step. Drop-in replacement for _update_nqs that uses
Stochastic Reconfiguration (diagonal Fisher natural gradient) instead of Adam.

Same teacher + energy + entropy loss as v1; only the optimizer changes.
"""
from __future__ import annotations

import numpy as np
import torch

from ._core import _ibm_format_to_configs
from ._nqs_sr_optimizer import sr_step


def update_nqs_sr(nqs, sr_lr, sr_damping, sr_fisher_K,
                  cumulative_bs, e0, sci_state, hamiltonian, cfg,
                  device, n_orb, n_qubits):
    """SR variant of _update_nqs.

    sr_lr: SR learning rate (typically 1e-3 ~ 1e-2)
    sr_damping: Fisher diagonal damping (typically 1e-4 ~ 1e-2)
    sr_fisher_K: number of samples for Fisher diagonal estimate (32-128)
    """
    configs = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
    n_total = len(configs)

    # Teacher weights (same as _update_nqs)
    amps = np.abs(sci_state.amplitudes) ** 2
    alpha_marginal = amps.sum(axis=1)
    beta_marginal = amps.sum(axis=0)
    alpha_marginal /= max(alpha_marginal.sum(), 1e-30)
    beta_marginal /= max(beta_marginal.sum(), 1e-30)

    alpha_map = {int(s): float(v) for s, v in zip(sci_state.ci_strs_a, alpha_marginal)}
    beta_map = {int(s): float(v) for s, v in zip(sci_state.ci_strs_b, beta_marginal)}

    powers_msb = (1 << np.arange(n_orb - 1, -1, -1)).astype(np.int64)
    bs_int = np.asarray(cumulative_bs).astype(np.int64)
    a_ints = (bs_int[:, :n_orb] * powers_msb).sum(axis=1)
    b_ints = (bs_int[:, n_orb:] * powers_msb).sum(axis=1)

    alpha_vals = np.fromiter(
        (alpha_map.get(int(a), 0.0) for a in a_ints),
        dtype=np.float64, count=n_total,
    )
    beta_vals = np.fromiter(
        (beta_map.get(int(b), 0.0) for b in b_ints),
        dtype=np.float64, count=n_total,
    )
    teacher_weights = alpha_vals * beta_vals
    total_w = teacher_weights.sum()
    if total_w > 0:
        teacher_weights /= total_w
    teacher_t = torch.from_numpy(teacher_weights).float().to(device)

    with torch.no_grad():
        diag_e = hamiltonian.diagonal_elements_batch(configs)
        if isinstance(diag_e, torch.Tensor):
            diag_e_t = diag_e.to(device=device, dtype=torch.float32)
        else:
            diag_e_t = torch.tensor(np.asarray(diag_e, dtype=np.float64),
                                    dtype=torch.float32, device=device)
        advantage = diag_e_t - e0

    max_batch = min(5000, n_total)

    for step in range(cfg.nqs_steps):
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

        # Closure capturing batch_teacher and batch_advantage
        def loss_fn(c):
            log_probs = nqs.log_prob(c)
            loss_teacher = -(batch_teacher * log_probs).sum()
            loss_energy = (batch_teacher * batch_advantage * log_probs).sum()
            loss_entropy = log_probs.mean()
            return (cfg.teacher_weight * loss_teacher
                    + cfg.energy_weight * loss_energy
                    + cfg.entropy_weight * loss_entropy)

        sr_step(nqs, batch_configs, loss_fn,
                lr=sr_lr, damping=sr_damping, fisher_K=sr_fisher_K)
