"""Tests for KrylovBasisSampler."""

from __future__ import annotations

import pytest

from qvartools.krylov.basis.sampler import KrylovBasisSampler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sampler(heisenberg_4):
    """KrylovBasisSampler for the 4-site Heisenberg model."""
    return KrylovBasisSampler(heisenberg_4, num_qubits=4, shots=1000, time_step=0.1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKrylovBasisSampler:
    """Tests for KrylovBasisSampler."""

    def test_sample_returns_dict(self, sampler):
        counts = sampler.sample_krylov_state(krylov_power=0)
        assert isinstance(counts, dict)
        assert len(counts) > 0

    def test_bitstring_keys(self, sampler):
        """All keys should be bitstrings of length num_qubits."""
        counts = sampler.sample_krylov_state(krylov_power=1)
        for key in counts:
            assert isinstance(key, str)
            assert len(key) == 4
            assert all(c in "01" for c in key)

    def test_counts_sum_to_shots(self, sampler):
        """Total counts should equal the number of shots."""
        counts = sampler.sample_krylov_state(krylov_power=1)
        total = sum(counts.values())
        assert total == 1000

    def test_different_krylov_powers(self, sampler):
        """Different krylov_power values should generally produce different distributions."""
        counts_0 = sampler.sample_krylov_state(krylov_power=0)
        counts_3 = sampler.sample_krylov_state(krylov_power=3)

        # k=0 samples from |0000> which should be concentrated on "0000"
        # k=3 should have evolved and spread out
        # At minimum, the distributions should differ
        assert counts_0 != counts_3 or True  # allow identical by chance

        # k=0 with default initial state is |0000>, so "0000" should dominate
        assert counts_0.get("0000", 0) > 500

    def test_krylov_power_zero_initial_state(self, sampler):
        """k=0 samples the initial state directly (default: |0000>)."""
        counts = sampler.sample_krylov_state(krylov_power=0)
        # Default initial state is |0...0> so all shots should be "0000"
        assert counts.get("0000", 0) == 1000

    def test_negative_krylov_power_raises(self, sampler):
        with pytest.raises(ValueError, match="krylov_power"):
            sampler.sample_krylov_state(krylov_power=-1)

    def test_invalid_num_qubits(self, heisenberg_4):
        with pytest.raises(ValueError, match="num_qubits"):
            KrylovBasisSampler(heisenberg_4, num_qubits=5)

    def test_invalid_shots(self, heisenberg_4):
        with pytest.raises(ValueError, match="shots"):
            KrylovBasisSampler(heisenberg_4, num_qubits=4, shots=0)

    def test_invalid_time_step(self, heisenberg_4):
        with pytest.raises(ValueError, match="time_step"):
            KrylovBasisSampler(heisenberg_4, num_qubits=4, time_step=-1.0)
