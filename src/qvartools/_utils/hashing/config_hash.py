"""
config_hash --- Overflow-safe integer hashing for binary configurations
=======================================================================

Binary configurations are encoded as integers for fast deduplication and
membership-checking. For n_sites >= 64, splits into two halves to avoid
int64 overflow.
"""

from __future__ import annotations

from typing import List, Union

import torch

__all__ = [
    "ConfigHash",
    "config_integer_hash",
]

ConfigHash = Union[int, tuple]


def config_integer_hash(
    configs: torch.Tensor,
) -> List[ConfigHash]:
    """Hash binary configuration vectors to integers.

    For n_sites < 64, each config is encoded as a single int64.
    For n_sites >= 64, the config is split into two halves returning
    a tuple ``(hash_hi, hash_lo)`` that is hashable and collision-free.

    Parameters
    ----------
    configs : torch.Tensor
        ``(n_configs, n_sites)`` binary tensor with values 0 or 1.

    Returns
    -------
    list
        Length-n_configs list of hashable elements (int or tuple of int).
    """
    if configs.numel() == 0:
        return []

    n_configs, n_sites = configs.shape
    device = configs.device

    if n_sites < 64:
        powers = (2 ** torch.arange(n_sites, device=device, dtype=torch.long)).flip(0)
        return (configs.long() * powers).sum(dim=1).cpu().tolist()
    else:
        half = n_sites // 2
        n_lo = n_sites - half

        powers_hi = (2 ** torch.arange(half, device=device, dtype=torch.long)).flip(0)
        powers_lo = (2 ** torch.arange(n_lo, device=device, dtype=torch.long)).flip(0)

        hash_hi = (configs[:, :half].long() * powers_hi).sum(dim=1)
        hash_lo = (configs[:, half:].long() * powers_lo).sum(dim=1)

        return list(zip(hash_hi.cpu().tolist(), hash_lo.cpu().tolist()))
