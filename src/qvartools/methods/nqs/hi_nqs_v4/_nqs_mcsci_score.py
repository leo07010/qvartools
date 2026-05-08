"""γ-MCSCI: Moment-Conditioned Selection CI score.

Replaces PT2 sum-rule with H-Krylov moment combination:
    score(σ') = w₁|⟨σ'|H|c⟩| + w₂|⟨σ'|H²|c⟩|/‖v₁‖ + w₃|⟨σ'|H³|c⟩|/‖v₂‖

where v_k = H^k c (Krylov vectors built by iterative matvec on basis).

H¹ alone is CIPSI/PT2 — first-order perturbation, sees dets that strongly
couple to a single basis det. H² and H³ pick up dets that contribute via
multi-step coupling — the "collective small-amplitude" tail that defines
dynamical correlation. Solanki/Ding/Reiher (arXiv:2602.12993) explicitly
admit NQS-SC cannot capture this; γ-MCSCI directly attacks it.

Differentiation: Krylov-style spectrum-aware selection, not PT2 first-order.
"""
from __future__ import annotations

import numpy as np
import torch


def _build_basis_amp_map(sci_state, n_orb):
    """Build {(α_int, β_int): c_σ} from sci_state.amplitudes."""
    amps = np.asarray(sci_state.amplitudes)
    ci_strs_a = np.asarray(sci_state.ci_strs_a)
    ci_strs_b = np.asarray(sci_state.ci_strs_b)
    nonzero = np.argwhere(np.abs(amps) > 1e-14)
    return {
        (int(ci_strs_a[ia]), int(ci_strs_b[ib])): float(amps[ia, ib])
        for ia, ib in nonzero
    }


def _basis_configs_from_sci_state(sci_state, n_orb, device):
    """Reconstruct basis configs (n_basis, 2*n_orb) NQS-format from sci_state."""
    amps = np.asarray(sci_state.amplitudes)
    ci_strs_a = np.asarray(sci_state.ci_strs_a)
    ci_strs_b = np.asarray(sci_state.ci_strs_b)
    nonzero = np.argwhere(np.abs(amps) > 1e-14)
    n_basis = len(nonzero)
    if n_basis == 0:
        return torch.zeros((0, 2 * n_orb), dtype=torch.long, device=device), \
               np.zeros(0, dtype=np.float64)
    configs = np.zeros((n_basis, 2 * n_orb), dtype=np.int64)
    coeffs = np.zeros(n_basis, dtype=np.float64)
    for k, (ia, ib) in enumerate(nonzero):
        a = int(ci_strs_a[ia]); b = int(ci_strs_b[ib])
        for j in range(n_orb):
            configs[k, j] = (a >> j) & 1
            configs[k, n_orb + j] = (b >> j) & 1
        coeffs[k] = float(amps[ia, ib])
    return torch.from_numpy(configs).long().to(device), coeffs


def _apply_H_to_vec(basis_configs, vec, hamiltonian, n_orb):
    """Compute H @ vec where vec is supported on basis_configs.

    Returns (out_configs_list, out_vec) where out_vec[i] = (H @ vec)[out_configs[i]]
    on the union basis ∪ S+D(basis). Output is sparse representation (only
    non-zero contributions).

    This is the Krylov matvec H · vec where vec lives on a CI-basis support.
    """
    device = hamiltonian.device
    n_basis = basis_configs.shape[0]
    if n_basis == 0:
        return torch.zeros((0, 2*n_orb), dtype=torch.long, device=device), \
               torch.zeros(0, dtype=torch.double, device=device)

    # Get all H-connections from basis
    connected, elements, src_idx = hamiltonian.get_connections_vectorized_batch(
        basis_configs.float()
    )
    if len(connected) == 0:
        return torch.zeros((0, 2*n_orb), dtype=torch.long, device=device), \
               torch.zeros(0, dtype=torch.double, device=device)

    # H_νσ × vec_σ (per pair); group by ν to sum
    vec_t = torch.as_tensor(vec, dtype=torch.double, device=device)
    contrib = elements.double() * vec_t[src_idx.long()]    # H_νσ * vec_σ

    # Group connected dets by unique key (a_int, b_int) and sum contributions
    powers = (2 ** torch.arange(n_orb, device=device, dtype=torch.long))
    a_ints = (connected[:, :n_orb].long() * powers).sum(dim=1)
    b_ints = (connected[:, n_orb:].long() * powers).sum(dim=1)
    pair_keys = a_ints * (2**n_orb) + b_ints   # composite key (assumes n_orb ≤ 31)

    # Also include diagonal contribution: vec_σ stays in result for σ ∈ basis
    # via H_σσ × vec_σ which is captured because diagonal IS included in get_connections
    # (or do we need to add it manually? — let's check)
    # Per molecular.py, get_connections_vectorized_batch returns OFF-diagonal only
    # So we must add diagonal: for σ in basis, contribution to (H·vec)[σ] += H_σσ·vec_σ
    diag_basis = hamiltonian.diagonal_elements_batch(basis_configs.float())
    diag_basis_t = torch.as_tensor(diag_basis, dtype=torch.double, device=device)
    diag_contrib = diag_basis_t * vec_t   # (H_σσ · vec_σ) for σ in basis

    # Build output: keys = unique connected configs ∪ basis configs
    basis_a = (basis_configs[:, :n_orb].long() * powers).sum(dim=1)
    basis_b = (basis_configs[:, n_orb:].long() * powers).sum(dim=1)
    basis_keys = basis_a * (2**n_orb) + basis_b

    all_keys = torch.cat([pair_keys, basis_keys])
    all_contrib = torch.cat([contrib, diag_contrib])
    all_a_int = torch.cat([a_ints, basis_a])
    all_b_int = torch.cat([b_ints, basis_b])

    # Sort + accumulate per unique key
    sorted_keys, sort_idx = torch.sort(all_keys)
    sorted_contrib = all_contrib[sort_idx]
    sorted_a = all_a_int[sort_idx]
    sorted_b = all_b_int[sort_idx]

    # Find unique boundaries
    unique_keys, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)
    n_unique = unique_keys.shape[0]

    out_vec = torch.zeros(n_unique, dtype=torch.double, device=device)
    out_vec.scatter_add_(0, inverse, sorted_contrib)

    # Reconstruct configs from unique a/b ints
    # Map unique_keys → first occurrence in sorted_a/sorted_b
    boundaries = torch.cat([torch.tensor([0], device=device, dtype=torch.long),
                            torch.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1])
    unique_a = sorted_a[boundaries]
    unique_b = sorted_b[boundaries]

    # Reconstruct config from a, b ints
    out_configs = torch.zeros((n_unique, 2 * n_orb), dtype=torch.long, device=device)
    for j in range(n_orb):
        out_configs[:, j] = (unique_a >> j) & 1
        out_configs[:, n_orb + j] = (unique_b >> j) & 1

    return out_configs, out_vec


def compute_mcsci_score(new_candidates, sci_state, hamiltonian, n_orb, n_qubits,
                       weights=(1.0, 0.5, 0.25), max_order=2):
    """Compute γ-MCSCI score for each candidate.

    score(σ') = Σ_k weights[k] · |⟨σ'|H^(k+1)|c⟩| / ‖v_k‖

    Args:
        new_candidates: list of (ibm_row, hash, config_tensor) tuples
        sci_state: prior diagonalization eigenvector
        hamiltonian: MolecularHamiltonian
        n_orb, n_qubits: integers
        weights: (w1, w2, w3) for H, H², H³ moments
        max_order: 1 = CIPSI baseline, 2 = +H², 3 = +H²+H³

    Returns:
        scores: ndarray of shape (n_candidates,)
    """
    device = hamiltonian.device
    if not new_candidates:
        return np.zeros(0)

    # Build initial basis vector v_0 = c
    basis_configs, c = _basis_configs_from_sci_state(sci_state, n_orb, device)
    if basis_configs.shape[0] == 0:
        return np.zeros(len(new_candidates))

    # Krylov moments: v_k = H^k @ c, supports grow each step
    moment_supports = []   # [(configs_k, vec_k)] for k = 1, 2, ..., max_order
    cur_configs = basis_configs
    cur_vec = torch.as_tensor(c, dtype=torch.double, device=device)
    for k in range(max_order):
        cur_configs, cur_vec = _apply_H_to_vec(cur_configs, cur_vec, hamiltonian, n_orb)
        # Normalize
        nrm = float(torch.norm(cur_vec).item())
        if nrm > 1e-14:
            moment_supports.append((cur_configs, cur_vec / nrm))
        else:
            moment_supports.append((cur_configs, cur_vec))

    # For each candidate σ', compute s_k(σ') = |⟨σ'|H|v_(k-1)⟩|
    # = Σ_µ H_σ'µ · v_(k-1)_µ for µ in support of v_(k-1)
    # This is essentially "evaluate v_k at σ'" since v_k = H · v_(k-1)
    cand_configs = torch.stack([c[2] for c in new_candidates]).to(device)
    n_cand = cand_configs.shape[0]

    scores = np.zeros(n_cand, dtype=np.float64)
    powers = (2 ** torch.arange(n_orb, device=device, dtype=torch.long))
    cand_a = (cand_configs[:, :n_orb].long() * powers).sum(dim=1)
    cand_b = (cand_configs[:, n_orb:].long() * powers).sum(dim=1)
    cand_keys = (cand_a * (2**n_orb) + cand_b).cpu().numpy()

    for k in range(max_order):
        # v_k support already computed, v_(k+1) = H·v_k. Score s_(k+1) for cand σ'
        # is just (H·v_k)[σ'] = v_(k+1)[σ'].
        # Compute v_(k+1) on full extended support and look up at cand_keys.
        v_configs, v_vec = _apply_H_to_vec(
            moment_supports[k][0], moment_supports[k][1], hamiltonian, n_orb)
        v_a = (v_configs[:, :n_orb].long() * powers).sum(dim=1)
        v_b = (v_configs[:, n_orb:].long() * powers).sum(dim=1)
        v_keys = (v_a * (2**n_orb) + v_b).cpu().numpy()
        v_vec_np = v_vec.cpu().numpy()

        # Build dict for fast lookup
        v_map = dict(zip(v_keys.tolist(), v_vec_np.tolist()))

        # Score contribution at each candidate
        for i in range(n_cand):
            v_at_cand = v_map.get(int(cand_keys[i]), 0.0)
            scores[i] += weights[k] * abs(v_at_cand)

    return scores
