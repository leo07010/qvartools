"""Tests for ConnectionCache."""

from __future__ import annotations

import torch
import pytest

from qvartools._utils.hashing.connection_cache import ConnectionCache


class TestConnectionCache:
    """Tests for ConnectionCache put, get, stats, eviction, etc."""

    @pytest.fixture()
    def cache(self) -> ConnectionCache:
        return ConnectionCache(max_size=5)

    @pytest.fixture()
    def config_a(self) -> torch.Tensor:
        return torch.tensor([1, 0, 1, 0])

    @pytest.fixture()
    def config_b(self) -> torch.Tensor:
        return torch.tensor([0, 1, 0, 1])

    def test_put_and_get_roundtrip(self, cache: ConnectionCache) -> None:
        config = torch.tensor([1, 0, 1, 0])
        connections = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0]])
        elements = torch.tensor([0.5, -0.3])

        cache.put(config, connections, elements)
        result = cache.get(config)

        assert result is not None
        retrieved_conn, retrieved_elem = result
        assert torch.equal(retrieved_conn, connections)
        assert torch.equal(retrieved_elem, elements)

    def test_cache_miss_returns_none(
        self, cache: ConnectionCache, config_a: torch.Tensor
    ) -> None:
        result = cache.get(config_a)
        assert result is None

    def test_stats_hits_misses(self, cache: ConnectionCache) -> None:
        config = torch.tensor([1, 0, 1, 0])
        connections = torch.tensor([[1, 1, 0, 0]])
        elements = torch.tensor([1.0])

        cache.put(config, connections, elements)
        cache.get(config)   # hit
        cache.get(config)   # hit
        cache.get(torch.tensor([0, 0, 0, 0]))  # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 2.0 / 3.0) < 1e-6

    def test_max_size_eviction(self) -> None:
        cache = ConnectionCache(max_size=2)
        dummy_conn = torch.tensor([[1, 0]])
        dummy_elem = torch.tensor([1.0])

        cache.put(torch.tensor([1, 0]), dummy_conn, dummy_elem)
        cache.put(torch.tensor([0, 1]), dummy_conn, dummy_elem)
        assert len(cache) == 2

        # Adding a third should evict the oldest
        cache.put(torch.tensor([1, 1]), dummy_conn, dummy_elem)
        assert len(cache) == 2

    def test_clear_empties_cache(self, cache: ConnectionCache) -> None:
        config = torch.tensor([1, 0, 1, 0])
        cache.put(config, torch.tensor([[0, 0, 0, 0]]), torch.tensor([0.0]))
        cache.get(config)

        cache.clear()
        assert len(cache) == 0
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_len(self, cache: ConnectionCache) -> None:
        assert len(cache) == 0
        cache.put(torch.tensor([1, 0, 0, 0]), torch.tensor([[0, 0, 0, 0]]), torch.tensor([0.0]))
        assert len(cache) == 1

    def test_contains(self, cache: ConnectionCache) -> None:
        config = torch.tensor([1, 0, 1, 0])
        assert config not in cache
        cache.put(config, torch.tensor([[0, 0, 0, 0]]), torch.tensor([0.0]))
        assert config in cache
