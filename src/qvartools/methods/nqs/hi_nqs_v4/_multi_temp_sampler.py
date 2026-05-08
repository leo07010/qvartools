"""Multi-temperature NQS sampling.

Replaces single-T forward sample with union of samples at K temperatures.
Each T captures different parts of |ψ_NQS|² distribution:
  - Low T (0.3): exploitation, peak / mode region
  - Medium T (1.0): balanced
  - High T (2.0+): exploration, long tail

Total n_samples is split evenly across K temperatures. Output is concatenated
config tensor + log_probs (same interface as nqs.sample).
"""
from __future__ import annotations
from typing import List
import torch


def install_multi_temp_sampler(temperatures: List[float]):
    """Patch AutoregressiveTransformer.sample to sample at multiple T's union.

    Returns the original sample method; pass it to uninstall_multi_temp_sampler
    to restore.
    """
    from ._transformer import AutoregressiveTransformer
    orig = AutoregressiveTransformer.sample
    _temps = list(temperatures)
    K = len(_temps)
    if K == 0:
        return orig

    @torch.no_grad()
    def _multi_t_sample(self, n_samples, hard=True, temperature=1.0):
        per_t = max(1, n_samples // K)
        cfgs, lps = [], []
        for t in _temps:
            c, lp = orig(self, per_t, hard=hard, temperature=float(t))
            cfgs.append(c)
            lps.append(lp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(cfgs, dim=0), torch.cat(lps, dim=0)

    AutoregressiveTransformer.sample = _multi_t_sample
    return orig


def uninstall_multi_temp_sampler(orig_method):
    from ._transformer import AutoregressiveTransformer
    AutoregressiveTransformer.__init__  # force module load
    AutoregressiveTransformer.sample = orig_method
