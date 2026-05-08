"""Multi-GPU NQS sampling (data-parallel).

Replicates the Transformer NQS across K visible GPUs. Each GPU samples
n_samples/K configs independently. Combined on CPU.

Why this works: autoregressive sampling is sequential WITHIN a sample
(token-by-token), but different samples are INDEPENDENT. So we can
trivially split the sample count across GPUs.

Weight sync: each GPU holds a replica of the NQS model. The master NQS
is the one that gets optimizer updates in _update_nqs. After each update
we copy the master's state_dict to all replicas before the next sample
call. Cost: ~120 MB per replica × (K-1) replicas at PCIe speed ~60 ms,
negligible vs sampling time.

Speedup: near-linear in K for the sampling phase. For a run where
sampling is 80% of wall time, 8 GPUs could drop total by ~4x.
"""
from __future__ import annotations

import concurrent.futures
import copy
from typing import List, Optional, Tuple

import torch


class MultiGPUNQSSampler:
    """Data-parallel sampler wrapping K replicas of an NQS model.

    Usage:
        sampler = MultiGPUNQSSampler(nqs_master, devices)
        configs, log_probs = sampler.sample(1_000_000, temperature=1.0)
        # ... train nqs_master ...
        sampler.sync_from_master(nqs_master)  # push updated weights to replicas
    """

    def __init__(self, nqs_master, devices: List[torch.device]):
        self.devices = devices
        self.K = len(devices)
        self.master = nqs_master  # on devices[0]
        self.replicas = [nqs_master]  # device 0 uses master directly
        for i in range(1, self.K):
            replica = copy.deepcopy(nqs_master).to(devices[i])
            replica.eval()
            self.replicas.append(replica)

    def sync_from_master(self, nqs_master=None):
        """Copy master's state_dict to all replicas (call after optimizer step)."""
        if nqs_master is None:
            nqs_master = self.master
        state = nqs_master.state_dict()
        for i in range(1, self.K):
            target_state = {
                k: v.to(self.devices[i], non_blocking=True) for k, v in state.items()
            }
            self.replicas[i].load_state_dict(target_state)

    @torch.no_grad()
    def sample(self, n_samples: int, hard: bool = True,
               temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n_samples configurations in parallel across K GPUs.

        Returns (configs, log_probs) on the MASTER device (devices[0])
        so downstream code sees same layout as single-GPU sample.
        """
        if self.K == 1 or n_samples <= self.K:
            cfg, lp = self.master.sample(n_samples, hard=hard, temperature=temperature)
            return cfg, lp

        # Partition n_samples evenly (+1 for first few GPUs if not divisible)
        chunks = [n_samples // self.K] * self.K
        for i in range(n_samples % self.K):
            chunks[i] += 1

        def _work(k):
            if chunks[k] == 0:
                return (k, None, None)
            torch.cuda.set_device(self.devices[k])
            cfg, lp = self.replicas[k].sample(
                chunks[k], hard=hard, temperature=temperature
            )
            # Move to CPU so main thread can concat across devices safely
            return (k, cfg.cpu(), lp.cpu())

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.K) as exe:
            futures = [exe.submit(_work, k) for k in range(self.K)]
            results = [f.result() for f in futures]

        results.sort(key=lambda r: r[0])
        cfgs = torch.cat([r[1] for r in results if r[1] is not None], dim=0)
        lps = torch.cat([r[2] for r in results if r[2] is not None], dim=0)
        # Move back to master device to match the single-GPU interface
        return cfgs.to(self.devices[0]), lps.to(self.devices[0])
