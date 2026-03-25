"""
connection_cache --- Hash-based cache for Hamiltonian connections
================================================================

Provides :class:`ConnectionCache`, a dictionary-backed cache that maps
configuration hashes to their Hamiltonian-connected configurations and
matrix elements.  This avoids redundant calls to
:meth:`Hamiltonian.get_connections` when the same configuration is
encountered multiple times during iterative basis expansion or sampling.

The hash function converts a binary occupation vector to a unique integer
by interpreting it as a binary number (via powers of 2).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

__all__ = [
    "ConnectionCache",
]

logger = logging.getLogger(__name__)


def _hash_config(config: torch.Tensor) -> int:
    """Convert a binary configuration tensor to a unique integer hash.

    The configuration is interpreted as a binary number using powers of 2
    (big-endian: site 0 is the most significant bit).

    Parameters
    ----------
    config : torch.Tensor
        Binary configuration vector, shape ``(n_sites,)`` with entries in
        ``{0, 1}``.

    Returns
    -------
    int
        Unique integer hash for the configuration.
    """
    n = config.shape[0]
    powers = torch.tensor(
        [1 << k for k in range(n - 1, -1, -1)],
        dtype=torch.int64,
        device=config.device,
    )
    return int((config.to(torch.int64) * powers).sum().item())


class ConnectionCache:
    """Hash-based cache for Hamiltonian connections.

    Stores ``(connected_configs, matrix_elements)`` tuples keyed by the
    integer hash of each configuration.  Provides O(1) lookup and
    automatic eviction when the cache exceeds ``max_size``.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of entries the cache may hold (default ``100000``).
        When the cache is full, the oldest entry is evicted on the next
        :meth:`put` call (FIFO eviction).

    Attributes
    ----------
    max_size : int
        Maximum cache capacity.

    Examples
    --------
    >>> cache = ConnectionCache(max_size=1000)
    >>> config = torch.tensor([1, 0, 1, 0])
    >>> cache.put(config, connected, elements)
    >>> result = cache.get(config)
    """

    def __init__(self, max_size: int = 100_000) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.max_size: int = max_size
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(
        self, config: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Look up cached connections for a configuration.

        Parameters
        ----------
        config : torch.Tensor
            Binary configuration vector, shape ``(n_sites,)``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor) or None
            ``(connected_configs, matrix_elements)`` if found, otherwise
            ``None``.
        """
        key = _hash_config(config)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(
        self,
        config: torch.Tensor,
        connections: torch.Tensor,
        elements: torch.Tensor,
    ) -> None:
        """Store connections for a configuration in the cache.

        If the cache is at capacity, the oldest entry is evicted (FIFO).

        Parameters
        ----------
        config : torch.Tensor
            Binary configuration vector, shape ``(n_sites,)``.
        connections : torch.Tensor
            Connected configurations, shape ``(n_conn, n_sites)``.
        elements : torch.Tensor
            Matrix elements, shape ``(n_conn,)``.
        """
        key = _hash_config(config)

        # Evict oldest entry if at capacity and key is new
        if key not in self._cache and len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = (connections.clone(), elements.clone())

    def clear(self) -> None:
        """Remove all entries from the cache and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"hits"`` : int --- number of successful lookups.
            - ``"misses"`` : int --- number of failed lookups.
            - ``"hit_rate"`` : float --- fraction of lookups that were hits
              (0.0 if no lookups have been made).
            - ``"size"`` : int --- current number of cached entries.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }

    def __len__(self) -> int:
        """Return the number of cached entries.

        Returns
        -------
        int
            Current cache size.
        """
        return len(self._cache)

    def __contains__(self, config: torch.Tensor) -> bool:
        """Check whether a configuration is in the cache.

        Parameters
        ----------
        config : torch.Tensor
            Binary configuration vector, shape ``(n_sites,)``.

        Returns
        -------
        bool
            ``True`` if the configuration hash is in the cache.
        """
        key = _hash_config(config)
        return key in self._cache

    def __repr__(self) -> str:
        return (
            f"ConnectionCache(size={len(self._cache)}, "
            f"max_size={self.max_size}, "
            f"hit_rate={self.stats()['hit_rate']:.2%})"
        )
