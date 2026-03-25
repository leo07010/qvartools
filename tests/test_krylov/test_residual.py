"""Tests for residual-based and selected-CI basis expansion."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qvartools.hamiltonians import HeisenbergHamiltonian
from qvartools.krylov.expansion.residual_expansion import (
    ResidualBasedExpander,
    ResidualExpansionConfig,
    SelectedCIExpander,
    _diagonalise_in_basis,
    _generate_candidate_configs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def heisenberg_4():
    """4-site periodic Heisenberg model."""
    return HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True)


@pytest.fixture()
def small_basis():
    """A small hand-picked basis of connected configurations for a 4-site system."""
    return torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.int64,
    )


@pytest.fixture()
def initial_eigenpair(heisenberg_4, small_basis):
    """Diagonalize in the small basis to get initial energy/eigenvector."""
    energy, eigenvector = _diagonalise_in_basis(heisenberg_4, small_basis)
    return energy, eigenvector


# ---------------------------------------------------------------------------
# Diagonalization helper
# ---------------------------------------------------------------------------


class TestDiagonaliseInBasis:
    """Tests for the _diagonalise_in_basis helper."""

    def test_returns_finite_energy(self, heisenberg_4, small_basis):
        energy, eigvec = _diagonalise_in_basis(heisenberg_4, small_basis)
        assert np.isfinite(energy)
        assert eigvec.shape == (small_basis.shape[0],)

    def test_full_basis_gives_exact_energy(self, heisenberg_4):
        """Diagonalizing in the full Hilbert space should give exact ground state."""
        all_configs = heisenberg_4._generate_all_configs()
        energy, _ = _diagonalise_in_basis(heisenberg_4, all_configs)
        exact_energy, _ = heisenberg_4.exact_ground_state()
        assert abs(energy - exact_energy) < 1e-6


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


class TestGenerateCandidateConfigs:
    """Tests for _generate_candidate_configs."""

    def test_candidates_not_in_basis(self, heisenberg_4, small_basis):
        candidates = _generate_candidate_configs(heisenberg_4, small_basis)
        basis_set = {tuple(small_basis[i].tolist()) for i in range(small_basis.shape[0])}
        for i in range(candidates.shape[0]):
            assert tuple(candidates[i].tolist()) not in basis_set

    def test_full_basis_no_candidates(self, heisenberg_4):
        """When basis contains all configs, no new candidates should appear."""
        all_configs = heisenberg_4._generate_all_configs()
        candidates = _generate_candidate_configs(heisenberg_4, all_configs)
        assert candidates.shape[0] == 0


# ---------------------------------------------------------------------------
# ResidualBasedExpander
# ---------------------------------------------------------------------------


class TestResidualBasedExpander:
    """Tests for ResidualBasedExpander."""

    def test_expand_basis_returns_larger(self, heisenberg_4, small_basis, initial_eigenpair):
        energy, eigvec = initial_eigenpair
        config = ResidualExpansionConfig(
            max_configs_per_iter=50,
            residual_threshold=1e-6,
            max_iterations=3,
            max_basis_size=16,
        )
        expander = ResidualBasedExpander(heisenberg_4, config)
        expanded, stats = expander.expand_basis(small_basis, energy, eigvec)

        assert expanded.shape[0] >= small_basis.shape[0]
        assert "iterations" in stats
        assert "final_energy" in stats

    def test_expanded_energy_variational(self, heisenberg_4, small_basis, initial_eigenpair):
        """Expanded basis energy must be <= initial energy (variational principle)."""
        energy, eigvec = initial_eigenpair
        config = ResidualExpansionConfig(
            max_configs_per_iter=50,
            residual_threshold=1e-6,
            max_iterations=5,
            max_basis_size=16,
        )
        expander = ResidualBasedExpander(heisenberg_4, config)
        expanded, stats = expander.expand_basis(small_basis, energy, eigvec)

        assert stats["final_energy"] <= energy + 1e-10

    def test_edge_case_full_basis(self, heisenberg_4):
        """When basis is already complete, expansion should not add configs."""
        all_configs = heisenberg_4._generate_all_configs()
        energy, eigvec = _diagonalise_in_basis(heisenberg_4, all_configs)

        config = ResidualExpansionConfig(
            max_configs_per_iter=50,
            residual_threshold=1e-6,
            max_iterations=3,
            max_basis_size=100,
        )
        expander = ResidualBasedExpander(heisenberg_4, config)
        expanded, stats = expander.expand_basis(all_configs, energy, eigvec)

        assert expanded.shape[0] == all_configs.shape[0]


# ---------------------------------------------------------------------------
# SelectedCIExpander
# ---------------------------------------------------------------------------


class TestSelectedCIExpander:
    """Tests for SelectedCIExpander (CIPSI-style)."""

    def test_expand_basis_returns_larger(self, heisenberg_4, small_basis, initial_eigenpair):
        energy, eigvec = initial_eigenpair
        config = ResidualExpansionConfig(
            max_configs_per_iter=50,
            residual_threshold=1e-8,
            max_iterations=3,
            max_basis_size=16,
        )
        expander = SelectedCIExpander(heisenberg_4, config)
        expanded, stats = expander.expand_basis(small_basis, energy, eigvec)

        assert expanded.shape[0] >= small_basis.shape[0]
        assert "pt2_corrections" in stats

    def test_expanded_energy_variational(self, heisenberg_4, small_basis, initial_eigenpair):
        """Expanded basis energy must be <= initial energy."""
        energy, eigvec = initial_eigenpair
        config = ResidualExpansionConfig(
            max_configs_per_iter=50,
            residual_threshold=1e-8,
            max_iterations=5,
            max_basis_size=16,
        )
        expander = SelectedCIExpander(heisenberg_4, config)
        _, stats = expander.expand_basis(small_basis, energy, eigvec)

        assert stats["final_energy"] <= energy + 1e-10

    def test_edge_case_full_basis(self, heisenberg_4):
        """No expansion when basis is already complete."""
        all_configs = heisenberg_4._generate_all_configs()
        energy, eigvec = _diagonalise_in_basis(heisenberg_4, all_configs)

        config = ResidualExpansionConfig(
            max_configs_per_iter=50,
            residual_threshold=1e-8,
            max_iterations=3,
            max_basis_size=100,
        )
        expander = SelectedCIExpander(heisenberg_4, config)
        expanded, stats = expander.expand_basis(all_configs, energy, eigvec)

        assert expanded.shape[0] == all_configs.shape[0]
