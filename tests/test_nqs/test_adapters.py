"""Tests for NQS adapter layers (ADR-001 Phase 2).

TDD: Tests written before implementation.
Verifies TransformerAsNQS and NQSWithSampling adapters.
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# TransformerAsNQS tests
# ---------------------------------------------------------------------------


class TestTransformerAsNQS:
    """TransformerAsNQS wraps AutoregressiveTransformer for NF training."""

    @pytest.fixture()
    def adapter(self):
        from qvartools.nqs.adapters import TransformerAsNQS
        from qvartools.nqs.transformer.autoregressive import (
            AutoregressiveTransformer,
        )

        transformer = AutoregressiveTransformer(
            n_orbitals=3, n_alpha=1, n_beta=1, embed_dim=16, n_heads=2, n_layers=1
        )
        return TransformerAsNQS(transformer)

    def test_importable(self):
        from qvartools.nqs.adapters import TransformerAsNQS

        assert TransformerAsNQS is not None

    def test_inherits_neural_quantum_state(self, adapter):
        from qvartools.nqs.neural_state import NeuralQuantumState

        assert isinstance(adapter, NeuralQuantumState)

    def test_has_num_sites(self, adapter):
        assert adapter.num_sites == 6  # 2 * n_orbitals = 2 * 3

    def test_log_amplitude_returns_correct_shape(self, adapter):
        x = torch.randint(0, 2, (4, 6), dtype=torch.float32)
        result = adapter.log_amplitude(x)
        assert result.shape == (4,)

    def test_log_amplitude_is_finite(self, adapter):
        x = torch.randint(0, 2, (8, 6), dtype=torch.float32)
        result = adapter.log_amplitude(x)
        assert torch.all(torch.isfinite(result))

    def test_phase_returns_zeros(self, adapter):
        x = torch.randint(0, 2, (4, 6), dtype=torch.float32)
        result = adapter.phase(x)
        assert result.shape == (4,)
        assert torch.allclose(result, torch.zeros(4))

    def test_complex_output_is_false(self, adapter):
        assert adapter.complex_output is False

    def test_gradient_flows_through(self, adapter):
        """Gradients must flow back to the transformer parameters."""
        x = torch.randint(0, 2, (4, 6), dtype=torch.float32)
        log_amp = adapter.log_amplitude(x)
        loss = log_amp.sum()
        loss.backward()
        # Check at least one transformer parameter got a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters()
        )
        assert has_grad

    def test_exported_from_nqs_package(self):
        from qvartools.nqs import TransformerAsNQS

        assert TransformerAsNQS is not None

    def test_wrong_input_dim_raises(self, adapter):
        """Wrong number of sites should cause an error in the transformer."""
        x = torch.randint(0, 2, (4, 3), dtype=torch.float32)  # 3 != 6
        with pytest.raises((RuntimeError, ValueError)):
            adapter.log_amplitude(x)

    def test_empty_batch(self, adapter):
        x = torch.zeros(0, 6, dtype=torch.float32)
        result = adapter.log_amplitude(x)
        assert result.shape == (0,)


# ---------------------------------------------------------------------------
# NQSWithSampling tests
# ---------------------------------------------------------------------------


class TestNQSWithSampling:
    """NQSWithSampling wraps any NeuralQuantumState for HI training."""

    @pytest.fixture()
    def adapter(self):
        from qvartools.nqs.adapters import NQSWithSampling
        from qvartools.nqs.architectures.dense import DenseNQS

        nqs = DenseNQS(num_sites=6, hidden_dims=[16, 8])
        return NQSWithSampling(nqs, n_orbitals=3, n_alpha=1, n_beta=1)

    def test_importable(self):
        from qvartools.nqs.adapters import NQSWithSampling

        assert NQSWithSampling is not None

    def test_sample_returns_correct_shape(self, adapter):
        configs = adapter.sample(10, temperature=1.0)
        assert configs.shape == (10, 6)

    def test_sample_is_binary(self, adapter):
        configs = adapter.sample(8)
        assert torch.all((configs == 0) | (configs == 1))

    def test_sample_conserves_particles(self, adapter):
        configs = adapter.sample(16)
        alpha = configs[:, :3]
        beta = configs[:, 3:]
        assert torch.all(alpha.sum(dim=1) == 1)  # n_alpha = 1
        assert torch.all(beta.sum(dim=1) == 1)  # n_beta = 1

    def test_log_prob_returns_correct_shape(self, adapter):
        alpha = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        beta = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        result = adapter.log_prob(alpha, beta)
        assert result.shape == (2,)

    def test_log_prob_is_finite(self, adapter):
        alpha = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        beta = torch.tensor([[0, 1, 0]], dtype=torch.float32)
        result = adapter.log_prob(alpha, beta)
        assert torch.all(torch.isfinite(result))

    def test_exported_from_nqs_package(self):
        from qvartools.nqs import NQSWithSampling

        assert NQSWithSampling is not None

    def test_large_system_raises_memory_guard(self):
        from qvartools.nqs.adapters import NQSWithSampling
        from qvartools.nqs.architectures.dense import DenseNQS

        nqs = DenseNQS(num_sites=40, hidden_dims=[16])
        adapter = NQSWithSampling(nqs, n_orbitals=20, n_alpha=10, n_beta=10)
        with pytest.raises(RuntimeError, match="enumeration"):
            adapter.sample(1)

    def test_log_prob_with_int_input(self, adapter):
        """log_prob should handle integer tensors via .float() conversion."""
        alpha = torch.tensor([[1, 0, 0]], dtype=torch.long)
        beta = torch.tensor([[0, 1, 0]], dtype=torch.long)
        result = adapter.log_prob(alpha, beta)
        assert torch.all(torch.isfinite(result))

    def test_sample_different_particle_counts(self):
        from qvartools.nqs.adapters import NQSWithSampling
        from qvartools.nqs.architectures.dense import DenseNQS

        nqs = DenseNQS(num_sites=8, hidden_dims=[16])
        adapter = NQSWithSampling(nqs, n_orbitals=4, n_alpha=2, n_beta=2)
        configs = adapter.sample(10)
        assert configs.shape == (10, 8)
        assert torch.all(configs[:, :4].sum(dim=1) == 2)
        assert torch.all(configs[:, 4:].sum(dim=1) == 2)
