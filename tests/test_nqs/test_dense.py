"""Tests for DenseNQS and SignedDenseNQS architectures."""

from __future__ import annotations

import torch
import pytest

from qvartools.nqs.architectures.dense import DenseNQS, SignedDenseNQS, compile_nqs


# ---------------------------------------------------------------------------
# DenseNQS
# ---------------------------------------------------------------------------


class TestDenseNQS:
    """Tests for DenseNQS forward pass and output shapes."""

    @pytest.fixture()
    def nqs(self) -> DenseNQS:
        return DenseNQS(num_sites=6, hidden_dims=[32, 16])

    @pytest.fixture()
    def batch(self) -> torch.Tensor:
        return torch.randint(0, 2, (8, 6)).float()

    def test_log_amplitude_shape(self, nqs: DenseNQS, batch: torch.Tensor) -> None:
        result = nqs.log_amplitude(batch)
        assert result.shape == (8,)

    def test_phase_shape(self, nqs: DenseNQS, batch: torch.Tensor) -> None:
        result = nqs.phase(batch)
        assert result.shape == (8,)

    def test_phase_is_zero_when_real(self, nqs: DenseNQS, batch: torch.Tensor) -> None:
        result = nqs.phase(batch)
        assert torch.allclose(result, torch.zeros(8))

    def test_log_psi_returns_tensor_for_real(
        self, nqs: DenseNQS, batch: torch.Tensor
    ) -> None:
        result = nqs.log_psi(batch)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (8,)

    def test_psi_shape(self, nqs: DenseNQS, batch: torch.Tensor) -> None:
        result = nqs.psi(batch)
        assert result.shape == (8,)

    def test_probability_shape(self, nqs: DenseNQS, batch: torch.Tensor) -> None:
        result = nqs.probability(batch)
        assert result.shape == (8,)

    def test_probability_non_negative(
        self, nqs: DenseNQS, batch: torch.Tensor
    ) -> None:
        result = nqs.probability(batch)
        assert (result >= 0).all()

    def test_probability_matches_exp_2_log_amplitude(
        self, nqs: DenseNQS, batch: torch.Tensor
    ) -> None:
        prob = nqs.probability(batch)
        log_amp = nqs.log_amplitude(batch)
        expected = torch.exp(2.0 * log_amp)
        assert torch.allclose(prob, expected, atol=1e-6)


class TestDenseNQSComplex:
    """Tests for DenseNQS with complex_output=True."""

    @pytest.fixture()
    def nqs(self) -> DenseNQS:
        return DenseNQS(num_sites=6, hidden_dims=[32, 16], complex_output=True)

    @pytest.fixture()
    def batch(self) -> torch.Tensor:
        return torch.randint(0, 2, (8, 6)).float()

    def test_phase_not_all_zero(self, nqs: DenseNQS, batch: torch.Tensor) -> None:
        """Complex NQS should produce non-zero phases for most inputs."""
        result = nqs.phase(batch)
        assert result.shape == (8,)
        # With random weights, it is extremely unlikely all phases are exactly zero
        assert not torch.allclose(result, torch.zeros(8), atol=1e-6)

    def test_log_psi_returns_tuple(self, nqs: DenseNQS, batch: torch.Tensor) -> None:
        result = nqs.log_psi(batch)
        assert isinstance(result, tuple)
        log_amp, phase = result
        assert log_amp.shape == (8,)
        assert phase.shape == (8,)

    def test_psi_is_complex(self, nqs: DenseNQS, batch: torch.Tensor) -> None:
        result = nqs.psi(batch)
        assert result.is_complex()


class TestDenseNQSBatch:
    """Tests for batch processing and edge cases."""

    def test_single_config(self) -> None:
        nqs = DenseNQS(num_sites=4, hidden_dims=[16])
        x = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
        log_amp = nqs.log_amplitude(x)
        assert log_amp.shape == (1,)

    def test_large_batch(self) -> None:
        nqs = DenseNQS(num_sites=4, hidden_dims=[16])
        x = torch.randint(0, 2, (512, 4)).float()
        log_amp = nqs.log_amplitude(x)
        assert log_amp.shape == (512,)

    def test_single_site(self) -> None:
        nqs = DenseNQS(num_sites=1, hidden_dims=[8])
        x = torch.tensor([[0], [1]], dtype=torch.float32)
        log_amp = nqs.log_amplitude(x)
        assert log_amp.shape == (2,)


# ---------------------------------------------------------------------------
# SignedDenseNQS
# ---------------------------------------------------------------------------


class TestSignedDenseNQS:
    """Tests for SignedDenseNQS."""

    @pytest.fixture()
    def nqs(self) -> SignedDenseNQS:
        return SignedDenseNQS(num_sites=6, hidden_dims=[32, 16])

    @pytest.fixture()
    def batch(self) -> torch.Tensor:
        return torch.randint(0, 2, (8, 6)).float()

    def test_log_amplitude_shape(
        self, nqs: SignedDenseNQS, batch: torch.Tensor
    ) -> None:
        result = nqs.log_amplitude(batch)
        assert result.shape == (8,)

    def test_phase_shape(self, nqs: SignedDenseNQS, batch: torch.Tensor) -> None:
        result = nqs.phase(batch)
        assert result.shape == (8,)

    def test_phase_is_zero_or_pi_in_mode(
        self, nqs: SignedDenseNQS, batch: torch.Tensor
    ) -> None:
        nqs.eval()
        result = nqs.phase(batch)
        # Each phase value should be either 0 or pi
        is_zero = torch.isclose(result, torch.zeros_like(result), atol=1e-6)
        is_pi = torch.isclose(result, torch.full_like(result, torch.pi), atol=1e-6)
        assert (is_zero | is_pi).all()

    def test_feature_caching(
        self, nqs: SignedDenseNQS, batch: torch.Tensor
    ) -> None:
        """Calling log_amplitude and phase with same tensor uses cache."""
        nqs.eval()
        _ = nqs.log_amplitude(batch)
        assert nqs._cached_input_id == id(batch)
        assert nqs._cached_features is not None

    def test_clear_feature_cache(
        self, nqs: SignedDenseNQS, batch: torch.Tensor
    ) -> None:
        nqs.eval()
        _ = nqs.log_amplitude(batch)
        nqs.clear_feature_cache()
        assert nqs._cached_input_id is None
        assert nqs._cached_features is None


# ---------------------------------------------------------------------------
# compile_nqs
# ---------------------------------------------------------------------------


class TestCompileNQS:
    """Tests for compile_nqs utility."""

    def test_compile_returns_module(self) -> None:
        nqs = DenseNQS(num_sites=4, hidden_dims=[16])
        compiled = compile_nqs(nqs)
        assert isinstance(compiled, torch.nn.Module)

    def test_compile_does_not_crash(self) -> None:
        nqs = DenseNQS(num_sites=4, hidden_dims=[16])
        # Should not raise regardless of platform support
        compiled = compile_nqs(nqs, mode="reduce-overhead")
        assert compiled is not None
