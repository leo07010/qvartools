"""Tests for ParticleConservingFlowSampler."""

from __future__ import annotations

import torch
import pytest

from qvartools.flows.networks.particle_conserving_flow import (
    ParticleConservingFlowSampler,
    verify_particle_conservation,
)


class TestParticleConservingFlowSampler:
    """Tests for particle-conserving flow sampling."""

    @pytest.fixture()
    def flow(self) -> ParticleConservingFlowSampler:
        return ParticleConservingFlowSampler(
            num_sites=10, n_alpha=2, n_beta=2, hidden_dims=[32, 16]
        )

    def test_sample_shape(self, flow: ParticleConservingFlowSampler) -> None:
        configs, unique = flow.sample(batch_size=32)
        assert configs.shape == (32, 10)
        assert unique.ndim == 2
        assert unique.shape[1] == 10

    def test_particle_conservation(
        self, flow: ParticleConservingFlowSampler
    ) -> None:
        configs, _ = flow.sample(batch_size=64)
        alpha_part = configs[:, :5]
        beta_part = configs[:, 5:]
        alpha_counts = alpha_part.sum(dim=1)
        beta_counts = beta_part.sum(dim=1)
        assert torch.allclose(alpha_counts, torch.full((64,), 2.0))
        assert torch.allclose(beta_counts, torch.full((64,), 2.0))

    def test_set_temperature(self, flow: ParticleConservingFlowSampler) -> None:
        flow.set_temperature(0.5)
        assert flow.temperature == 0.5
        assert flow.selector.temperature == 0.5

    def test_set_temperature_clamps_to_min(
        self, flow: ParticleConservingFlowSampler
    ) -> None:
        flow.set_temperature(0.001)
        assert flow.temperature == flow.min_temperature

    def test_sample_without_replacement_unique(
        self, flow: ParticleConservingFlowSampler
    ) -> None:
        configs = flow.sample_without_replacement(batch_size=16)
        # All rows should be unique
        unique = torch.unique(configs, dim=0)
        assert unique.shape[0] == configs.shape[0]

    def test_different_particle_numbers(self) -> None:
        flow = ParticleConservingFlowSampler(
            num_sites=8, n_alpha=1, n_beta=3, hidden_dims=[16]
        )
        configs, _ = flow.sample(batch_size=32)
        alpha_counts = configs[:, :4].sum(dim=1)
        beta_counts = configs[:, 4:].sum(dim=1)
        assert torch.allclose(alpha_counts, torch.full((32,), 1.0))
        assert torch.allclose(beta_counts, torch.full((32,), 3.0))


class TestVerifyParticleConservation:
    """Tests for verify_particle_conservation."""

    def test_valid_configs(self) -> None:
        configs = torch.tensor([
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        ], dtype=torch.float32)
        is_valid, stats = verify_particle_conservation(
            configs, n_orbitals=5, n_alpha=2, n_beta=2
        )
        assert is_valid
        assert stats["n_valid"] == 2
        assert stats["n_invalid"] == 0

    def test_invalid_configs(self) -> None:
        configs = torch.tensor([
            [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],  # 3 alpha instead of 2
        ], dtype=torch.float32)
        is_valid, stats = verify_particle_conservation(
            configs, n_orbitals=5, n_alpha=2, n_beta=2
        )
        assert not is_valid
        assert stats["alpha_violations"] == 1

    def test_wrong_shape_raises(self) -> None:
        configs = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        with pytest.raises(ValueError, match="2-dimensional"):
            verify_particle_conservation(configs, n_orbitals=2, n_alpha=1, n_beta=1)

    def test_wrong_columns_raises(self) -> None:
        configs = torch.tensor([[1, 0, 1]], dtype=torch.float32)
        with pytest.raises(ValueError, match="columns"):
            verify_particle_conservation(configs, n_orbitals=2, n_alpha=1, n_beta=1)
