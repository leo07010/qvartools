"""Tests for ComplexNQS and RBMQuantumState architectures."""

from __future__ import annotations

import torch
import pytest

from qvartools.nqs.architectures.complex_nqs import ComplexNQS, RBMQuantumState


# ---------------------------------------------------------------------------
# ComplexNQS
# ---------------------------------------------------------------------------


class TestComplexNQS:
    """Tests for ComplexNQS."""

    @pytest.fixture()
    def nqs(self) -> ComplexNQS:
        return ComplexNQS(num_sites=6, hidden_dims=[32, 16])

    @pytest.fixture()
    def batch(self) -> torch.Tensor:
        return torch.randint(0, 2, (8, 6)).float()

    def test_log_amplitude_shape(self, nqs: ComplexNQS, batch: torch.Tensor) -> None:
        result = nqs.log_amplitude(batch)
        assert result.shape == (8,)

    def test_phase_shape(self, nqs: ComplexNQS, batch: torch.Tensor) -> None:
        result = nqs.phase(batch)
        assert result.shape == (8,)

    def test_phase_is_nonzero(self, nqs: ComplexNQS, batch: torch.Tensor) -> None:
        """With random weights, phase should not be all zeros."""
        result = nqs.phase(batch)
        assert not torch.allclose(result, torch.zeros(8), atol=1e-6)

    def test_phase_bounded(self, nqs: ComplexNQS, batch: torch.Tensor) -> None:
        """Phase should be in (-pi, pi) due to tanh scaling."""
        result = nqs.phase(batch)
        assert (result > -torch.pi).all()
        assert (result < torch.pi).all()

    def test_log_psi_returns_tuple(self, nqs: ComplexNQS, batch: torch.Tensor) -> None:
        result = nqs.log_psi(batch)
        assert isinstance(result, tuple)
        log_amp, phase = result
        assert log_amp.shape == (8,)
        assert phase.shape == (8,)

    def test_feature_caching(self, nqs: ComplexNQS, batch: torch.Tensor) -> None:
        _ = nqs.log_amplitude(batch)
        assert nqs._cached_input_id == id(batch)
        _ = nqs.phase(batch)
        # Same tensor so cache should be reused (no recomputation)
        assert nqs._cached_input_id == id(batch)

    def test_clear_feature_cache(self, nqs: ComplexNQS, batch: torch.Tensor) -> None:
        _ = nqs.log_amplitude(batch)
        nqs.clear_feature_cache()
        assert nqs._cached_input_id is None
        assert nqs._cached_features is None


# ---------------------------------------------------------------------------
# RBMQuantumState
# ---------------------------------------------------------------------------


class TestRBMQuantumState:
    """Tests for RBMQuantumState."""

    @pytest.fixture()
    def rbm_real(self) -> RBMQuantumState:
        return RBMQuantumState(num_sites=6, num_hidden=10, complex_weights=False)

    @pytest.fixture()
    def rbm_complex(self) -> RBMQuantumState:
        return RBMQuantumState(num_sites=6, num_hidden=10, complex_weights=True)

    @pytest.fixture()
    def batch(self) -> torch.Tensor:
        return torch.randint(0, 2, (8, 6)).float()

    def test_log_amplitude_shape(
        self, rbm_real: RBMQuantumState, batch: torch.Tensor
    ) -> None:
        result = rbm_real.log_amplitude(batch)
        assert result.shape == (8,)

    def test_phase_shape(
        self, rbm_real: RBMQuantumState, batch: torch.Tensor
    ) -> None:
        result = rbm_real.phase(batch)
        assert result.shape == (8,)

    def test_real_rbm_phase_is_zero(
        self, rbm_real: RBMQuantumState, batch: torch.Tensor
    ) -> None:
        result = rbm_real.phase(batch)
        assert torch.allclose(result, torch.zeros(8))

    def test_complex_rbm_log_amplitude_shape(
        self, rbm_complex: RBMQuantumState, batch: torch.Tensor
    ) -> None:
        result = rbm_complex.log_amplitude(batch)
        assert result.shape == (8,)

    def test_complex_rbm_phase_nontrivial(
        self, rbm_complex: RBMQuantumState, batch: torch.Tensor
    ) -> None:
        """Complex RBM should produce non-zero phases."""
        result = rbm_complex.phase(batch)
        assert result.shape == (8,)
        assert not torch.allclose(result, torch.zeros(8), atol=1e-6)

    def test_numerical_stability_no_nan(
        self, rbm_real: RBMQuantumState, batch: torch.Tensor
    ) -> None:
        log_amp = rbm_real.log_amplitude(batch)
        assert not torch.isnan(log_amp).any()
        assert not torch.isinf(log_amp).any()

    def test_numerical_stability_complex_no_nan(
        self, rbm_complex: RBMQuantumState, batch: torch.Tensor
    ) -> None:
        log_amp = rbm_complex.log_amplitude(batch)
        assert not torch.isnan(log_amp).any()
        assert not torch.isinf(log_amp).any()

    def test_default_num_hidden(self) -> None:
        rbm = RBMQuantumState(num_sites=8)
        assert rbm.num_hidden == 8
