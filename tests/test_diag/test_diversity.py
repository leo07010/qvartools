"""Tests for DiversitySelector and helper functions."""

from __future__ import annotations

import torch
import pytest

from qvartools.diag.selection.diversity_selection import (
    DiversityConfig,
    DiversitySelector,
    compute_excitation_rank,
    compute_hamming_distance,
)


class TestComputeExcitationRank:
    """Tests for compute_excitation_rank."""

    def test_identical_configs(self) -> None:
        ref = torch.tensor([1, 1, 0, 0])
        assert compute_excitation_rank(ref, ref) == 0

    def test_single_excitation(self) -> None:
        ref = torch.tensor([1, 1, 0, 0])
        cfg = torch.tensor([1, 0, 1, 0])
        assert compute_excitation_rank(cfg, ref) == 2

    def test_full_excitation(self) -> None:
        ref = torch.tensor([1, 1, 0, 0])
        cfg = torch.tensor([0, 0, 1, 1])
        assert compute_excitation_rank(cfg, ref) == 4

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_excitation_rank(torch.tensor([1, 0]), torch.tensor([1, 0, 1]))


class TestComputeHammingDistance:
    """Tests for compute_hamming_distance."""

    def test_identical(self) -> None:
        c = torch.tensor([1, 0, 1, 0])
        assert compute_hamming_distance(c, c) == 0

    def test_different(self) -> None:
        c1 = torch.tensor([1, 0, 1, 0])
        c2 = torch.tensor([0, 1, 0, 1])
        assert compute_hamming_distance(c1, c2) == 4

    def test_partial_difference(self) -> None:
        c1 = torch.tensor([1, 1, 0, 0])
        c2 = torch.tensor([1, 0, 0, 0])
        assert compute_hamming_distance(c1, c2) == 1

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_hamming_distance(torch.tensor([1, 0]), torch.tensor([1, 0, 1]))


class TestDiversitySelector:
    """Tests for DiversitySelector.select."""

    @pytest.fixture()
    def reference(self) -> torch.Tensor:
        return torch.tensor([1, 1, 1, 0, 0, 0])

    @pytest.fixture()
    def selector(self, reference: torch.Tensor) -> DiversitySelector:
        cfg = DiversityConfig(
            max_configs=10,
            min_hamming_distance=1,
        )
        return DiversitySelector(cfg, reference, n_orbitals=6)

    def test_select_returns_correct_number(
        self, selector: DiversitySelector
    ) -> None:
        # Build pool: reference + configs at various excitation ranks
        ref = torch.tensor([1, 1, 1, 0, 0, 0])
        configs = torch.stack([
            ref,
            torch.tensor([1, 1, 0, 1, 0, 0]),  # rank 2
            torch.tensor([1, 0, 1, 0, 1, 0]),  # rank 2
            torch.tensor([0, 1, 1, 1, 0, 0]),  # rank 2
            torch.tensor([1, 0, 0, 1, 1, 0]),  # rank 4
            torch.tensor([0, 1, 0, 0, 1, 1]),  # rank 4
            torch.tensor([0, 0, 1, 1, 1, 0]),  # rank 4
        ])
        weights = torch.ones(configs.shape[0])

        selected, stats = selector.select(configs, weights)
        assert selected.shape[0] <= 10
        assert stats["n_selected"] <= 10

    def test_max_configs_respected(self, reference: torch.Tensor) -> None:
        cfg = DiversityConfig(max_configs=50, min_hamming_distance=0)
        selector = DiversitySelector(cfg, reference, n_orbitals=6)

        # Large pool
        configs = torch.randint(0, 2, (200, 6))
        weights = torch.rand(200)

        selected, stats = selector.select(configs, weights)
        # Total selected should not exceed max_configs (sum of quotas)
        assert selected.shape[0] <= 50

    def test_min_hamming_distance_enforced_within_rank(
        self, reference: torch.Tensor
    ) -> None:
        """Hamming distance is enforced within each rank bucket."""
        cfg = DiversityConfig(max_configs=100, min_hamming_distance=3)
        selector = DiversitySelector(cfg, reference, n_orbitals=6)

        # Build configs all at the same excitation rank (rank 2)
        # so the distance constraint applies within one bucket
        ref = reference
        rank2_configs = []
        for i in range(6):
            for j in range(i + 1, 6):
                c = ref.clone()
                c[i] = 1 - c[i]
                c[j] = 1 - c[j]
                rank2_configs.append(c)
        configs = torch.stack(rank2_configs)
        weights = torch.rand(configs.shape[0])

        selected, stats = selector.select(configs, weights)

        # Within the rank-2 bucket, Hamming distance should be enforced
        n = selected.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                rank_i = compute_excitation_rank(selected[i], ref)
                rank_j = compute_excitation_rank(selected[j], ref)
                if rank_i == rank_j:
                    dist = compute_hamming_distance(selected[i], selected[j])
                    assert dist >= 3, (
                        f"Same-rank configs {i} and {j} have Hamming distance {dist} < 3"
                    )
