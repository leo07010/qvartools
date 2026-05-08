"""VMC + supervised hybrid update for NQS in HI+NQS+SQD.

Replaces the broken `_update_nqs` (alpha-beta marginal factorization +
basis-only supervised loss) with two correct learning signals:

  Supervised (Fix B): joint c² teacher
      Train log p_NQS(σ) ≈ log c²_σ on basis dets, using the JOINT amplitude
      from diagonalization (not the marginal product α·β which destroys
      α-β correlation).

  VMC (Fix A): coupling-importance REINFORCE on fresh NQS samples
      Sample σ ~ p_NQS, compute coupling
          K(σ) = Σ_{σ'∈basis} ⟨σ|H|σ'⟩ · c_{σ'}
      Treat |K(σ)|² as a positive importance score (proportional to the
      EN-PT2 contribution of σ to lowering E). REINFORCE-style gradient
          ∂_θ E ≈ -⟨ |K(σ)|² · ∂_θ log p_NQS(σ) ⟩
      pushes p_NQS UP for σ with strong off-diagonal connection to current
      eigenvector — i.e. the externals that would most reduce energy if
      added to the basis.

Why coupling² instead of textbook (E_loc - E):
  Autoregressive Transformer NQS represents only |ψ|² (no sign). E_loc
  computation requires ψ(σ') for σ' connected to σ — for σ' external the
  sign is unknown. Coupling K(σ) is signed (uses c_σ' from diag), but |K|²
  is sign-free and is exactly the numerator of the EN-PT2 score, so this
  surrogate is well-founded.

Energy reporting in the outer pipeline is unchanged: subspace
diagonalization gives E_diag, this update only changes how NQS is trained.
"""
from __future__ import annotations

import numpy as np
import torch

from ._core import _ibm_format_to_configs


def vmc_update_nqs(
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
    """Hybrid VMC + joint-supervised update."""
    n_basis = len(cumulative_bs)
    if n_basis == 0:
        return

    # -----------------------------------------------------------------
    # Joint c² teacher (Fix B) — supersedes alpha×beta marginal version
    # -----------------------------------------------------------------
    amps = np.asarray(sci_state.amplitudes)  # (na, nb), signed real
    ci_strs_a = np.asarray(sci_state.ci_strs_a)
    ci_strs_b = np.asarray(sci_state.ci_strs_b)

    # MSB-first bit packing matches what _update_nqs uses
    powers_msb = (1 << np.arange(n_orb - 1, -1, -1)).astype(np.int64)
    bs_int = np.asarray(cumulative_bs).astype(np.int64)
    a_ints_basis = (bs_int[:, :n_orb] * powers_msb).sum(axis=1)
    b_ints_basis = (bs_int[:, n_orb:] * powers_msb).sum(axis=1)

    a_to_ia = {int(s): i for i, s in enumerate(ci_strs_a)}
    b_to_ib = {int(s): i for i, s in enumerate(ci_strs_b)}

    joint_c = np.zeros(n_basis, dtype=np.float64)
    for k in range(n_basis):
        ia = a_to_ia.get(int(a_ints_basis[k]), -1)
        ib = b_to_ib.get(int(b_ints_basis[k]), -1)
        if ia >= 0 and ib >= 0:
            joint_c[k] = amps[ia, ib]

    teacher_w = joint_c ** 2
    total_w = teacher_w.sum()
    if total_w > 0:
        teacher_w /= total_w
    teacher_t = torch.from_numpy(teacher_w).float().to(device)

    # Configs in NQS occupation format (LSB-first within each spin sector)
    configs = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)

    # -----------------------------------------------------------------
    # Build basis amplitude map keyed by (alpha_int_LSB, beta_int_LSB)
    # to match `get_connections_vectorized_batch` output encoding.
    # `ci_strs_a/b` are MSB-first (matching cumulative_bs); `connected`
    # tensor uses LSB-first. We rebuild the map with LSB encoding.
    # -----------------------------------------------------------------
    powers_lsb = (1 << np.arange(n_orb)).astype(np.int64)
    # Per-basis-det LSB ints (recompute since ci_strs_a/b are MSB)
    a_lsb = (bs_int[:, :n_orb] * powers_lsb).sum(axis=1)
    b_lsb = (bs_int[:, n_orb:] * powers_lsb).sum(axis=1)
    coupling_map = {}
    for k in range(n_basis):
        c = float(joint_c[k])
        if abs(c) > 1e-14:
            coupling_map[(int(a_lsb[k]), int(b_lsb[k]))] = c

    # -----------------------------------------------------------------
    # Update loop — α·supervised + β·VMC + γ·entropy
    # -----------------------------------------------------------------
    n_vmc = int(getattr(cfg, "vmc_n_samples", 4096))
    vmc_weight = float(getattr(cfg, "vmc_weight", 1.0))
    teacher_weight = float(cfg.teacher_weight)
    entropy_weight = float(cfg.entropy_weight)
    max_batch = min(5000, n_basis)

    n_alpha_target = int(hamiltonian.n_alpha)
    n_beta_target = int(hamiltonian.n_beta)

    for step in range(cfg.nqs_steps):
        optimizer.zero_grad()

        # --- supervised on basis (Fix B: joint teacher) ---
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

        # --- VMC: sample, compute |coupling|², REINFORCE ---
        vmc_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            sampled, _ = nqs.sample(n_vmc, hard=True, temperature=1.0)
            sampled_long = sampled.long().cpu()
            n_a_actual = sampled_long[:, :n_orb].sum(dim=1)
            n_b_actual = sampled_long[:, n_orb:].sum(dim=1)
            valid_mask = (n_a_actual == n_alpha_target) & (n_b_actual == n_beta_target)
            valid_sampled = sampled[valid_mask].long()

        if valid_sampled.shape[0] > 0:
            with torch.no_grad():
                connected, elements, batch_idx = (
                    hamiltonian.get_connections_vectorized_batch(
                        valid_sampled.float().to(device)
                    )
                )
                if connected.shape[0] > 0:
                    powers_lsb_t = torch.tensor(powers_lsb, device=device, dtype=torch.long)
                    conn_a = (connected[:, :n_orb].long() * powers_lsb_t).sum(dim=1)
                    conn_b = (connected[:, n_orb:].long() * powers_lsb_t).sum(dim=1)
                    conn_a_np = conn_a.cpu().numpy()
                    conn_b_np = conn_b.cpu().numpy()
                    conn_c = np.fromiter(
                        (coupling_map.get((int(a), int(b)), 0.0)
                         for a, b in zip(conn_a_np, conn_b_np)),
                        dtype=np.float64, count=len(conn_a_np),
                    )
                    conn_c_t = torch.from_numpy(conn_c).to(device)
                    weighted = elements.double() * conn_c_t
                    coupling = torch.zeros(
                        valid_sampled.shape[0], dtype=torch.double, device=device,
                    )
                    coupling.scatter_add_(0, batch_idx.long(), weighted)
                    importance = (coupling ** 2).float()
                else:
                    importance = torch.zeros(valid_sampled.shape[0], device=device)

            # Normalise advantage so its scale doesn't dominate sup_loss
            mean_imp = importance.mean().item()
            if mean_imp > 1e-30:
                adv = (importance / mean_imp).detach()
                log_p_vmc = nqs.log_prob(valid_sampled.float().to(device))
                vmc_loss = -(adv * log_p_vmc).mean()

        # --- entropy regulariser (encourages mode coverage) ---
        entropy_loss = log_p_sup.mean()

        loss = (
            teacher_weight * sup_loss
            + vmc_weight * vmc_loss
            + entropy_weight * entropy_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
        optimizer.step()


def install_vmc_update_in_v3():
    """Monkey-patch v3's `_update_nqs` import in v3 module to use VMC version."""
    import src.methods.hi_nqs_sqd_v3 as _v3
    import src.methods.hi_nqs_sqd as _base
    _v3._update_nqs = vmc_update_nqs
    _base._update_nqs = vmc_update_nqs  # in case v3 imported it by reference
    print("[vmc_update] installed: hybrid joint-supervised + VMC reinforce", flush=True)
