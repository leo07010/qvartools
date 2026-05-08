"""MCMC sampling for NQS — Metropolis-Hastings using NQS as wave function.

Replaces the autoregressive `nqs.sample()` with an MH chain that targets the
true |ψ_NQS|² distribution. Properly explores the long tail (low-amp dets that
forward sampling can't easily reach), which is key for breaking the
classical_expansion plateau on hard systems like 52Q.

Move kernel: random single or double excitation that preserves N_alpha, N_beta.
  - 70% single alpha or beta excitation (occ→vir swap within one spin channel)
  - 30% double excitation (two simultaneous swaps)

Batched K parallel chains, each step does one NQS forward pass on (K, n_qubits).
Burn-in + sample collection. Optional replica exchange (skipped here for
simplicity; can be added later).
"""
from __future__ import annotations

import math
from typing import Optional

import torch


@torch.no_grad()
def _propose_batched_singles(configs: torch.Tensor, n_orb: int) -> torch.Tensor:
    """Propose K single excitations in batch. configs: (K, n_qubits) long.

    For each chain, randomly choose alpha or beta channel, then propose
    occ→vir swap within it. Particle number is preserved.
    """
    K = configs.shape[0]
    device = configs.device
    new_configs = configs.clone()

    # Random spin choice per chain (alpha or beta)
    is_alpha = torch.rand(K, device=device) < 0.5
    offset = torch.where(is_alpha, torch.zeros(K, dtype=torch.long, device=device),
                         torch.full((K,), n_orb, dtype=torch.long, device=device))

    for k in range(K):
        ofs = int(offset[k].item())
        spin_block = configs[k, ofs:ofs + n_orb]
        occ = torch.where(spin_block == 1)[0]
        vir = torch.where(spin_block == 0)[0]
        if len(occ) == 0 or len(vir) == 0:
            continue
        i = occ[torch.randint(len(occ), (1,), device=device).item()].item()
        a = vir[torch.randint(len(vir), (1,), device=device).item()].item()
        new_configs[k, ofs + i] = 0
        new_configs[k, ofs + a] = 1

    return new_configs


@torch.no_grad()
def _propose_batched_doubles(configs: torch.Tensor, n_orb: int) -> torch.Tensor:
    """Propose K double excitations in batch (two single swaps composed)."""
    out = _propose_batched_singles(configs, n_orb)
    out = _propose_batched_singles(out, n_orb)
    return out


@torch.no_grad()
def _propose_mixed(configs: torch.Tensor, n_orb: int,
                   double_frac: float = 0.3) -> torch.Tensor:
    """Mixed proposal: per chain, pick single or double move."""
    K = configs.shape[0]
    is_double = torch.rand(K, device=configs.device) < double_frac
    out_single = _propose_batched_singles(configs, n_orb)
    out_double = _propose_batched_doubles(configs, n_orb)
    return torch.where(is_double.unsqueeze(1), out_double, out_single)


@torch.no_grad()
def mcmc_sample(
    nqs,
    n_samples: int,
    n_orb: int,
    n_chains: int = 1024,
    n_burnin: int = 200,
    double_frac: float = 0.3,
    init_configs: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    return_log_probs: bool = False,
):
    """Metropolis-Hastings sampling from |ψ_NQS|² with K parallel chains.

    Returns (configs, log_probs) compatible with nqs.sample() interface.

    Steps per chain: n_burnin + n_samples / n_chains (rounded up).
    Total samples returned: ≥ n_samples.
    """
    device = next(nqs.parameters()).device

    # Initial chain states
    if init_configs is None:
        # Sample initial from NQS forward (cheap warm start)
        init_configs, _ = nqs.sample(n_chains, hard=True, temperature=temperature)
    elif init_configs.shape[0] < n_chains:
        # Replicate to fill K chains
        reps = (n_chains + init_configs.shape[0] - 1) // init_configs.shape[0]
        init_configs = init_configs.repeat(reps, 1)[:n_chains]

    configs = init_configs.long().to(device)

    # Initial log_probs
    log_p = nqs.log_prob(configs.float()) / max(temperature, 1e-6)

    n_total_steps = n_burnin + max(1, (n_samples + n_chains - 1) // n_chains)
    collected = []
    collected_lp = []
    accept_count = 0
    total_count = 0

    for step in range(n_total_steps):
        # Propose
        configs_new = _propose_mixed(configs, n_orb, double_frac=double_frac)
        log_p_new = nqs.log_prob(configs_new.float()) / max(temperature, 1e-6)

        # MH acceptance: target ∝ |ψ|² so log α = 2 (log_p_new - log_p)
        log_ratio = 2.0 * (log_p_new - log_p)
        accept_prob = torch.minimum(torch.ones_like(log_ratio),
                                     torch.exp(log_ratio.clamp(max=0)))
        accept = torch.rand(n_chains, device=device) < accept_prob

        # Update
        configs = torch.where(accept.unsqueeze(1), configs_new, configs)
        log_p = torch.where(accept, log_p_new, log_p)

        accept_count += int(accept.sum().item())
        total_count += n_chains

        if step >= n_burnin:
            collected.append(configs.clone())
            if return_log_probs:
                collected_lp.append(log_p.clone())

    samples = torch.cat(collected, dim=0)
    # Trim to exactly n_samples (or keep all)
    if samples.shape[0] > n_samples:
        idx = torch.randperm(samples.shape[0], device=device)[:n_samples]
        samples = samples[idx]
        if return_log_probs and collected_lp:
            lps = torch.cat(collected_lp, dim=0)[idx]
            return samples, lps
    elif return_log_probs and collected_lp:
        lps = torch.cat(collected_lp, dim=0)
        return samples, lps

    # Caller (v3 main loop) discards log_probs anyway — return placeholder zeros
    # to avoid materialising a (n_samples, n_qubits) → (n_samples,) batch forward
    # that would OOM on big basis (e.g. 80k samples through 256/8/8 Transformer).
    placeholder = torch.zeros(samples.shape[0], dtype=torch.float32,
                               device=samples.device)
    return samples, placeholder


def install_mcmc_sampler(n_chains: int = 1024, n_burnin: int = 200,
                         double_frac: float = 0.3):
    """Monkey-patch AutoregressiveTransformer.sample to use MCMC.

    Captures the original autoregressive sample method as `_forward_sample`
    so MCMC can use it for initial chain states without recursion.
    """
    from ._transformer import AutoregressiveTransformer
    orig = AutoregressiveTransformer.sample

    @torch.no_grad()
    def _mcmc_sample_method(self, n_samples, hard=True, temperature=1.0):
        n_orb = self.n_orbitals
        # Use ORIGINAL forward sample for initial chain states (no recursion).
        init_configs, _ = orig(self, n_chains, hard=hard, temperature=temperature)
        return mcmc_sample(
            self, n_samples=n_samples, n_orb=n_orb,
            n_chains=n_chains, n_burnin=n_burnin,
            double_frac=double_frac, temperature=temperature,
            init_configs=init_configs,
        )

    AutoregressiveTransformer.sample = _mcmc_sample_method
    return orig


def uninstall_mcmc_sampler(orig_method):
    """Restore original sample method."""
    from ._transformer import AutoregressiveTransformer
    AutoregressiveTransformer.sample = orig_method
