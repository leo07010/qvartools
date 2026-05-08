"""Stochastic Reconfiguration (SR) optimizer for NQS — Carleo & Troyer 2017 style.

For Transformer NQS at 15M params, exact SR (full Fisher matrix S = E[O O^T])
is infeasible (15M × 15M). We use the diagonal Fisher approximation:

    F_ii ≈ E_{x ~ NQS}[(∂_θᵢ log p_NQS(x))²]

Update rule:
    Δθ = -η · g / √(F_diag + λ)

where g is the standard loss gradient, η is learning rate, λ is damping.

This is closer to true SR than Adam (which uses per-step gradient EMA, not the
Fisher info of the *wavefunction distribution*). For NQS escaping mode collapse,
the Fisher-info preconditioner reweights updates so low-amp parameters get
proportionally more push — exactly what we want when the network is biased
to high-amp regions.

Per-sample log_prob gradients are computed via a Python loop over K configs
(K=64 default). Cost: K × backward = ~64 × 3ms = 200ms per SR step.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn


@torch.no_grad()
def _zero_grads(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def estimate_fisher_diag(model: nn.Module, configs: torch.Tensor,
                         K: int = 64) -> Dict[str, torch.Tensor]:
    """Estimate diagonal Fisher info F_ii = E[(∂_θᵢ log p)²].

    Loops over K samples (subsampled from configs), backward on each
    log_prob, accumulates squared gradients.
    """
    K = min(K, configs.shape[0])
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    # Subsample K configs
    if configs.shape[0] > K:
        idx = torch.randperm(configs.shape[0])[:K]
        sample = configs[idx]
    else:
        sample = configs

    for i in range(K):
        _zero_grads(model)
        # Forward + backward of single config's log_prob
        log_p = model.log_prob(sample[i:i + 1])[0]
        log_p.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2

    # Average
    fisher = {name: f / K for name, f in fisher.items()}
    return fisher


def sr_step(
    model: nn.Module,
    configs: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    lr: float = 1e-3,
    damping: float = 1e-4,
    fisher_K: int = 64,
    grad_clip: float = 1.0,
) -> float:
    """One SR update step.

    Steps:
      1. Estimate diagonal Fisher info from K sampled log_prob gradients.
      2. Compute full loss gradient via standard backward.
      3. Update: θ -= lr · g / √(F_diag + λ).
    """
    # Step 1: Fisher diagonal
    fisher_diag = estimate_fisher_diag(model, configs, K=fisher_K)

    # Step 2: total loss gradient
    _zero_grads(model)
    loss = loss_fn(configs)
    loss.backward()

    # Optional grad clip
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    # Step 3: natural gradient update
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            f = fisher_diag.get(name)
            if f is None:
                p.data -= lr * p.grad
                continue
            preconditioner = 1.0 / (f + damping).sqrt()
            p.data -= lr * preconditioner * p.grad

    return float(loss.detach().item())


class SROptimizer:
    """Drop-in replacement for torch.optim.Adam in NQS training loop.

    Wraps sr_step to provide a simpler API:

        opt = SROptimizer(model, lr=1e-3, damping=1e-4, fisher_K=64)
        for step in range(N):
            loss = opt.step_with(configs, loss_fn)
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3,
                 damping: float = 1e-4, fisher_K: int = 64,
                 grad_clip: float = 1.0):
        self.model = model
        self.lr = lr
        self.damping = damping
        self.fisher_K = fisher_K
        self.grad_clip = grad_clip

    def step_with(self, configs: torch.Tensor,
                  loss_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
        return sr_step(
            self.model, configs, loss_fn,
            lr=self.lr, damping=self.damping,
            fisher_K=self.fisher_K, grad_clip=self.grad_clip,
        )
