"""Tests for format utility functions."""

from __future__ import annotations

import torch

from qvartools._utils.formatting.bitstring_format import hash_config, vectorized_dedup


class TestHashConfig:
    """Tests for hash_config."""

    def test_consistent_hash(self) -> None:
        config = torch.tensor([1, 0, 1, 0])
        h1 = hash_config(config)
        h2 = hash_config(config)
        assert h1 == h2

    def test_known_value(self) -> None:
        # [1, 0, 1, 0] -> 1*8 + 0*4 + 1*2 + 0*1 = 10
        config = torch.tensor([1, 0, 1, 0])
        assert hash_config(config) == 10

    def test_all_ones(self) -> None:
        # [1, 1, 1, 1] -> 15
        config = torch.tensor([1, 1, 1, 1])
        assert hash_config(config) == 15

    def test_all_zeros(self) -> None:
        config = torch.tensor([0, 0, 0, 0])
        assert hash_config(config) == 0

    def test_different_configs_different_hashes(self) -> None:
        c1 = torch.tensor([1, 0, 0])
        c2 = torch.tensor([0, 1, 0])
        assert hash_config(c1) != hash_config(c2)


class TestVectorizedDedup:
    """Tests for vectorized_dedup."""

    def test_removes_duplicates(self) -> None:
        existing = torch.tensor([[1, 0, 1], [0, 1, 0]])
        new = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0]])
        result = vectorized_dedup(existing, new)
        assert result.shape == (1, 3)
        assert torch.equal(result[0], torch.tensor([1, 1, 0]))

    def test_no_overlap_returns_all_unique(self) -> None:
        existing = torch.tensor([[1, 0, 0]])
        new = torch.tensor([[0, 1, 0], [0, 0, 1]])
        result = vectorized_dedup(existing, new)
        assert result.shape[0] == 2

    def test_full_overlap_returns_empty(self) -> None:
        existing = torch.tensor([[1, 0, 1], [0, 1, 0]])
        new = torch.tensor([[1, 0, 1], [0, 1, 0]])
        result = vectorized_dedup(existing, new)
        assert result.shape[0] == 0

    def test_empty_new_returns_empty(self) -> None:
        existing = torch.tensor([[1, 0]])
        new = torch.zeros(0, 2, dtype=torch.long)
        result = vectorized_dedup(existing, new)
        assert result.shape[0] == 0

    def test_empty_existing_deduplicates_new(self) -> None:
        existing = torch.zeros(0, 3, dtype=torch.long)
        new = torch.tensor([[1, 0, 1], [1, 0, 1], [0, 1, 0]])
        result = vectorized_dedup(existing, new)
        assert result.shape[0] == 2
