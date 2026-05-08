"""Multi-GPU NQS sampling via data parallelism.

Replicates the NQS model across K visible GPUs. Each `nqs.sample(N)` call
splits N into K chunks, each GPU samples independently in a thread, results
are gathered on the master device.

Sync strategy:
  - Replicas are deep-copied on first call.
  - Before each sampling call, master's state_dict is broadcast to replicas
    (NQS weights change each training iter). Cost per sync ≈ 15M * 4 bytes
    per replica = 60 MB, tiny compared to ~200s sampling work.

Safe to enable/disable at runtime via install/uninstall.

Usage (installed via hi_nqs_sqd_v4 when use_multi_gpu_sampling=True):
    _prev = install_multi_gpu_sampling(devices, batch_per_gpu=50_000)
    # ... run the HI+NQS+SQD loop; every nqs.sample() dispatches to K GPUs
    uninstall_multi_gpu_sampling(_prev)
"""
from __future__ import annotations

import concurrent.futures
import copy
from typing import Callable, List, Optional

import torch

from ._transformer import AutoregressiveTransformer


# module-level state for install/uninstall
_replicas_cache: dict = {}            # id(master_nqs) -> list of replicas
_active_devices: List[torch.device] = []
_orig_sample: Optional[Callable] = None


def install_multi_gpu_sampling(devices: List[torch.device], batch_per_gpu: int = 50_000):
    """Patch AutoregressiveTransformer.sample with multi-GPU dispatcher.

    Returns (orig_sample, prev_active_devices) for uninstallation.
    """
    global _orig_sample, _active_devices
    if _orig_sample is None:
        _orig_sample = AutoregressiveTransformer.sample
    prev_devices = list(_active_devices)
    _active_devices = list(devices)

    @torch.no_grad()
    def _mg_sample(self, n_samples, hard=True, temperature=1.0):
        K = len(_active_devices)
        if K <= 1:
            # Single GPU: fall back to original (but still respect batching)
            return _batched_sample_single(self, n_samples, hard, temperature, batch_per_gpu)

        master_id = id(self)
        if master_id not in _replicas_cache:
            replicas = []
            for d in _active_devices:
                try:
                    mp = next(self.parameters()).device
                except StopIteration:
                    mp = torch.device("cpu")
                if mp == d:
                    replicas.append(self)
                else:
                    r = copy.deepcopy(self).to(d)
                    r.eval()
                    replicas.append(r)
            _replicas_cache[master_id] = replicas
        replicas = _replicas_cache[master_id]

        # Sync master weights to all replicas (weights change every iter)
        master_sd = self.state_dict()
        for r, d in zip(replicas, _active_devices):
            if r is self:
                continue
            r.load_state_dict({k: v.to(d, non_blocking=True) for k, v in master_sd.items()})

        # Split n_samples across K GPUs
        base = n_samples // K
        rem = n_samples % K
        splits = [base + (1 if i < rem else 0) for i in range(K)]

        def _work(k):
            n_k = splits[k]
            if n_k <= 0:
                return None
            torch.cuda.set_device(_active_devices[k].index)
            rep = replicas[k]
            # Batch within this GPU if n_k > batch_per_gpu
            if n_k <= batch_per_gpu:
                c, lp = _orig_sample(rep, n_k, hard=hard, temperature=temperature)
            else:
                cs, lps = [], []
                for s in range(0, n_k, batch_per_gpu):
                    e = min(s + batch_per_gpu, n_k)
                    c, lp = _orig_sample(rep, e - s, hard=hard, temperature=temperature)
                    cs.append(c)
                    lps.append(lp)
                    torch.cuda.empty_cache()
                c = torch.cat(cs, dim=0)
                lp = torch.cat(lps, dim=0)
            return (c.to(_active_devices[0], non_blocking=True),
                    lp.to(_active_devices[0], non_blocking=True))

        with concurrent.futures.ThreadPoolExecutor(max_workers=K) as exe:
            futures = [exe.submit(_work, k) for k in range(K)]
            results = [f.result() for f in futures]

        cfgs = [r[0] for r in results if r is not None]
        lps = [r[1] for r in results if r is not None]
        if not cfgs:
            return _orig_sample(self, 0, hard=hard, temperature=temperature)
        return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)

    AutoregressiveTransformer.sample = _mg_sample
    return (_orig_sample, prev_devices)


def uninstall_multi_gpu_sampling(state):
    """Restore original sample method and clear replica cache."""
    global _orig_sample, _active_devices, _replicas_cache
    orig, prev_devices = state
    AutoregressiveTransformer.sample = orig
    _active_devices = prev_devices
    _replicas_cache.clear()
    _orig_sample = None


@torch.no_grad()
def _batched_sample_single(nqs, n_samples, hard, temperature, batch_size):
    """Fall-back: single-GPU batched sampling."""
    if _orig_sample is None:
        # Caller forgot to install; use class method directly
        return AutoregressiveTransformer.sample(nqs, n_samples, hard=hard, temperature=temperature)
    if n_samples <= batch_size:
        return _orig_sample(nqs, n_samples, hard=hard, temperature=temperature)
    cfgs, lps = [], []
    for s in range(0, n_samples, batch_size):
        e = min(s + batch_size, n_samples)
        c, lp = _orig_sample(nqs, e - s, hard=hard, temperature=temperature)
        cfgs.append(c); lps.append(lp)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)
