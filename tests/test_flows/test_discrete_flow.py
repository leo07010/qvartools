"""Tests for DiscreteFlowSampler."""

from __future__ import annotations

import torch
import pytest

from qvartools.flows.networks.discrete_flow import DiscreteFlowSampler


class TestDiscreteFlowSampler:
    """Tests for DiscreteFlowSampler sampling and probability methods."""

    @pytest.fixture()
    def flow(self) -> DiscreteFlowSampler:
        return DiscreteFlowSampler(
            num_sites=6, num_coupling_layers=2, hidden_dims=[32], n_mc_samples=10
        )

    def test_sample_continuous_shape(self, flow: DiscreteFlowSampler) -> None:
        y = flow.sample_continuous(batch_size=16)
        assert y.shape == (16, 6)

    def test_sample_continuous_bounded(self, flow: DiscreteFlowSampler) -> None:
        y = flow.sample_continuous(batch_size=64)
        assert (y >= -1.0).all()
        assert (y <= 1.0).all()

    def test_discretize_produces_binary(self) -> None:
        y = torch.tensor([[-0.5, 0.3, 0.0, -0.1, 0.9, -0.9]])
        result = DiscreteFlowSampler.discretize(y)
        # Values >= 0 -> 1, values < 0 -> 0
        expected = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
        assert torch.equal(result, expected)

    def test_discretize_only_binary_values(self, flow: DiscreteFlowSampler) -> None:
        y = flow.sample_continuous(batch_size=32)
        discrete = flow.discretize(y)
        unique_vals = torch.unique(discrete)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_sample_returns_correct_shapes(self, flow: DiscreteFlowSampler) -> None:
        configs, unique_configs = flow.sample(batch_size=32)
        assert configs.shape == (32, 6)
        assert unique_configs.ndim == 2
        assert unique_configs.shape[1] == 6
        assert unique_configs.shape[0] <= 32

    def test_unique_is_subset_of_all(self, flow: DiscreteFlowSampler) -> None:
        configs, unique_configs = flow.sample(batch_size=64)
        # Every unique config should appear in the full configs set
        for i in range(unique_configs.shape[0]):
            found = (configs == unique_configs[i]).all(dim=1).any()
            assert found

    def test_log_prob_continuous_finite(self, flow: DiscreteFlowSampler) -> None:
        y = flow.sample_continuous(batch_size=16)
        log_prob = flow.log_prob_continuous(y)
        assert log_prob.shape == (16,)
        assert torch.isfinite(log_prob).all()

    def test_log_prob_discrete_finite(self, flow: DiscreteFlowSampler) -> None:
        configs, _ = flow.sample(batch_size=16)
        log_prob = flow.log_prob_discrete(configs)
        assert log_prob.shape == (16,)
        assert torch.isfinite(log_prob).all()

    def test_forward_returns_three_tensors(self, flow: DiscreteFlowSampler) -> None:
        configs, unique, log_probs = flow(batch_size=16)
        assert configs.shape == (16, 6)
        assert unique.ndim == 2
        assert log_probs.shape == (16,)
