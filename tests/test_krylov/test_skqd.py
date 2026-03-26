"""Tests for Sample-Based Krylov Quantum Diagonalization (SKQD)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qvartools.krylov.basis.skqd import (
    FlowGuidedSKQD,
    SampleBasedKrylovDiagonalization,
    SKQDConfig,
)

# ---------------------------------------------------------------------------
# SKQDConfig validation
# ---------------------------------------------------------------------------


class TestSKQDConfig:
    """Tests for SKQDConfig dataclass validation."""

    def test_valid_config(self):
        cfg = SKQDConfig(max_krylov_dim=5, time_step=0.1, shots_per_krylov=500)
        assert cfg.max_krylov_dim == 5
        assert cfg.time_step == 0.1
        assert cfg.shots_per_krylov == 500

    def test_invalid_max_krylov_dim(self):
        with pytest.raises(ValueError, match="max_krylov_dim"):
            SKQDConfig(max_krylov_dim=0)

    def test_invalid_time_step(self):
        with pytest.raises(ValueError, match="time_step"):
            SKQDConfig(time_step=-0.1)

    def test_invalid_shots(self):
        with pytest.raises(ValueError, match="shots_per_krylov"):
            SKQDConfig(shots_per_krylov=0)

    def test_invalid_num_eigenvalues(self):
        with pytest.raises(ValueError, match="num_eigenvalues"):
            SKQDConfig(num_eigenvalues=0)

    def test_invalid_regularization(self):
        with pytest.raises(ValueError, match="regularization"):
            SKQDConfig(regularization=-1.0)

    def test_config_is_frozen(self):
        cfg = SKQDConfig()
        with pytest.raises(AttributeError):
            cfg.max_krylov_dim = 20  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SampleBasedKrylovDiagonalization
# ---------------------------------------------------------------------------


class TestSKQD:
    """Tests for SampleBasedKrylovDiagonalization on a 4-site Heisenberg model."""

    def test_runs_without_error(self, heisenberg_4):
        config = SKQDConfig(max_krylov_dim=3, time_step=0.1, shots_per_krylov=200)
        skqd = SampleBasedKrylovDiagonalization(heisenberg_4, config)
        eigenvalues, info = skqd.run()
        assert eigenvalues is not None
        assert info is not None

    def test_energy_is_finite(self, heisenberg_4):
        config = SKQDConfig(max_krylov_dim=3, time_step=0.1, shots_per_krylov=500)
        skqd = SampleBasedKrylovDiagonalization(heisenberg_4, config)
        eigenvalues, info = skqd.run()
        assert np.isfinite(eigenvalues[0])

    def test_energy_below_max_diagonal(self, heisenberg_4):
        """Ground-state energy must be below the maximum diagonal element."""
        config = SKQDConfig(max_krylov_dim=3, time_step=0.1, shots_per_krylov=500)
        skqd = SampleBasedKrylovDiagonalization(heisenberg_4, config)
        eigenvalues, _ = skqd.run()

        # Compute max diagonal element
        all_configs = heisenberg_4._generate_all_configs()
        diag_elements = [
            float(heisenberg_4.diagonal_element(all_configs[i]))
            for i in range(all_configs.shape[0])
        ]
        max_diag = max(diag_elements)

        assert eigenvalues[0] <= max_diag + 1e-10

    def test_different_krylov_dims(self, heisenberg_4):
        """Larger Krylov dimension should generally give equal or better energy."""
        config_small = SKQDConfig(
            max_krylov_dim=2, time_step=0.1, shots_per_krylov=1000
        )
        config_large = SKQDConfig(
            max_krylov_dim=5, time_step=0.1, shots_per_krylov=1000
        )

        skqd_small = SampleBasedKrylovDiagonalization(heisenberg_4, config_small)
        skqd_large = SampleBasedKrylovDiagonalization(heisenberg_4, config_large)

        ev_small, _ = skqd_small.run()
        ev_large, _ = skqd_large.run()

        # Larger Krylov space should give energy <= small + some tolerance
        # (stochastic sampling means we allow a small tolerance)
        assert ev_large[0] <= ev_small[0] + 0.5

    def test_result_dict_keys(self, heisenberg_4):
        config = SKQDConfig(max_krylov_dim=3, time_step=0.1, shots_per_krylov=200)
        skqd = SampleBasedKrylovDiagonalization(heisenberg_4, config)
        _, info = skqd.run()

        expected_keys = {
            "basis_size",
            "krylov_dim",
            "energies_per_step",
            "basis_configs",
        }
        assert expected_keys == set(info.keys())
        assert info["krylov_dim"] == 3
        assert len(info["energies_per_step"]) == 3
        assert isinstance(info["basis_configs"], torch.Tensor)
        assert info["basis_size"] > 0


# ---------------------------------------------------------------------------
# FlowGuidedSKQD
# ---------------------------------------------------------------------------


class TestFlowGuidedSKQD:
    """Tests for FlowGuidedSKQD with NF basis seeding."""

    def test_run_with_nf(self, heisenberg_4):
        """FlowGuidedSKQD should run and return expected keys."""
        config = SKQDConfig(max_krylov_dim=2, time_step=0.1, shots_per_krylov=200)

        # Create a small NF basis (random binary configs)
        nf_basis = torch.randint(0, 2, (8, 4), dtype=torch.int64)

        skqd = FlowGuidedSKQD(heisenberg_4, config, nf_basis=nf_basis)
        results = skqd.run_with_nf()

        expected_keys = {
            "energy",
            "eigenvalues",
            "basis_size",
            "krylov_dim",
            "energies_per_step",
            "nf_energy",
            "basis_configs",
        }
        assert expected_keys == set(results.keys())
        assert np.isfinite(results["energy"])
        assert results["basis_size"] > 0

    def test_nf_basis_validation(self, heisenberg_4):
        """FlowGuidedSKQD should reject invalid NF basis shapes."""
        config = SKQDConfig()

        # Wrong number of sites
        with pytest.raises(ValueError, match="num_sites"):
            FlowGuidedSKQD(
                heisenberg_4, config, nf_basis=torch.zeros(5, 3, dtype=torch.int64)
            )

        # 1-D tensor
        with pytest.raises(ValueError, match="2-D"):
            FlowGuidedSKQD(
                heisenberg_4, config, nf_basis=torch.zeros(4, dtype=torch.int64)
            )
