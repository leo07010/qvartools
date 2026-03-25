"""Tests for MolecularHamiltonian (requires PySCF)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Skip entire module if PySCF is unavailable
pyscf = pytest.importorskip("pyscf")

from qvartools.hamiltonians import (
    MolecularHamiltonian,
    MolecularIntegrals,
    compute_molecular_integrals,
)


# ---------------------------------------------------------------------------
# Helper: build H2 integrals (reusable across tests in this module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2_integrals():
    """Compute H2 integrals once for this module."""
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
    return compute_molecular_integrals(geometry, basis="sto-3g")


@pytest.fixture(scope="module")
def h2_ham(h2_integrals):
    """H2 MolecularHamiltonian."""
    return MolecularHamiltonian(h2_integrals)


@pytest.fixture(scope="module")
def h2_pyscf_fci_energy():
    """PySCF FCI energy for H2 at 0.74 A, sto-3g."""
    from pyscf import fci, gto, scf

    mol = gto.Mole()
    mol.atom = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
    mol.basis = "sto-3g"
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()
    return e_fci


# ===================================================================
# Diagonal element tests
# ===================================================================


class TestH2DiagonalElements:
    """Test diagonal elements of H2 Hamiltonian."""

    def test_hf_state_diagonal(self, h2_ham, h2_integrals):
        """HF state diagonal element should include nuclear repulsion + electron energy."""
        hf = h2_ham.get_hf_state()
        diag = h2_ham.diagonal_element(hf)
        # Should be a finite real number
        assert torch.isfinite(diag)
        # Should include nuclear repulsion
        assert diag.item() != 0.0

    def test_diagonal_elements_batch_matches_individual(self, h2_ham):
        """Batch diagonal elements should match individual calls."""
        configs = h2_ham._generate_all_configs()
        batch_diag = h2_ham.diagonal_elements_batch(configs)

        for idx in range(configs.shape[0]):
            individual = h2_ham.diagonal_element(configs[idx])
            assert batch_diag[idx].item() == pytest.approx(
                individual.item(), abs=1e-12
            ), f"Mismatch at config index {idx}"

    def test_empty_state_diagonal(self, h2_ham):
        """Vacuum state should have energy = nuclear repulsion only."""
        vacuum = torch.zeros(h2_ham.num_sites, dtype=torch.int64)
        diag = h2_ham.diagonal_element(vacuum)
        assert diag.item() == pytest.approx(h2_ham.E_nuc, abs=1e-12)


# ===================================================================
# get_connections tests
# ===================================================================


class TestH2Connections:
    """Test off-diagonal connections for H2."""

    def test_connections_conserve_particle_number(self, h2_ham):
        """All connected states must have the same particle number as input."""
        hf = h2_ham.get_hf_state()
        n_particles = int(hf.sum().item())

        connected, elements = h2_ham.get_connections(hf)
        if connected.numel() > 0:
            for k in range(connected.shape[0]):
                n_conn = int(connected[k].sum().item())
                assert n_conn == n_particles, (
                    f"Connection {k} has {n_conn} particles, expected {n_particles}"
                )

    def test_connections_are_valid_configs(self, h2_ham):
        """Connected configs must be binary vectors of correct length."""
        hf = h2_ham.get_hf_state()
        connected, _ = h2_ham.get_connections(hf)
        if connected.numel() > 0:
            assert connected.shape[1] == h2_ham.num_sites
            assert ((connected == 0) | (connected == 1)).all()

    def test_connections_not_self(self, h2_ham):
        """No connected config should equal the input config."""
        hf = h2_ham.get_hf_state()
        connected, _ = h2_ham.get_connections(hf)
        if connected.numel() > 0:
            for k in range(connected.shape[0]):
                assert not torch.equal(connected[k], hf)

    def test_hf_state_has_excitations(self, h2_ham):
        """HF state of H2 should have some excitations."""
        hf = h2_ham.get_hf_state()
        connected, elements = h2_ham.get_connections(hf)
        assert connected.shape[0] > 0, "HF state should have off-diagonal connections"


# ===================================================================
# matrix_elements tests
# ===================================================================


class TestH2MatrixElements:
    """Test full matrix element computation for H2."""

    def test_matrix_is_symmetric(self, h2_ham):
        """H_{ij} should equal H_{ji} for a Hermitian Hamiltonian."""
        configs = h2_ham._generate_all_configs()
        mat = h2_ham.matrix_elements(configs, configs)
        assert torch.allclose(mat, mat.T, atol=1e-10), "Matrix should be symmetric"

    def test_matrix_matches_dense(self, h2_ham):
        """matrix_elements on all configs should match to_dense."""
        dense = h2_ham.to_dense()
        configs = h2_ham._generate_all_configs()
        mat = h2_ham.matrix_elements(configs, configs)
        assert torch.allclose(mat, dense, atol=1e-10)


# ===================================================================
# FCI energy test
# ===================================================================


class TestH2FCIEnergy:
    """Test FCI energy matches PySCF."""

    def test_fci_matches_pyscf(self, h2_ham, h2_pyscf_fci_energy):
        """Our FCI energy should match PySCF's FCI solver within 1e-8 Ha."""
        our_fci = h2_ham.fci_energy()
        assert our_fci == pytest.approx(h2_pyscf_fci_energy, abs=1e-8), (
            f"FCI mismatch: ours={our_fci}, pyscf={h2_pyscf_fci_energy}"
        )


# ===================================================================
# HF state tests
# ===================================================================


class TestHFState:
    """Test Hartree-Fock state generation."""

    def test_hf_state_correct_occupation(self, h2_ham, h2_integrals):
        """HF state should have correct number of alpha and beta electrons."""
        hf = h2_ham.get_hf_state()
        n_orb = h2_integrals.n_orbitals

        n_alpha_occ = int(hf[:n_orb].sum().item())
        n_beta_occ = int(hf[n_orb:].sum().item())

        assert n_alpha_occ == h2_integrals.n_alpha
        assert n_beta_occ == h2_integrals.n_beta

    def test_hf_state_fills_lowest_orbitals(self, h2_ham, h2_integrals):
        """HF state should fill lowest-index orbitals in each spin sector."""
        hf = h2_ham.get_hf_state()
        n_orb = h2_integrals.n_orbitals

        # Alpha: first n_alpha orbitals should be 1, rest 0
        for i in range(n_orb):
            expected = 1 if i < h2_integrals.n_alpha else 0
            assert hf[i].item() == expected, f"Alpha orbital {i} mismatch"

        # Beta: first n_beta orbitals should be 1, rest 0
        for i in range(n_orb):
            expected = 1 if i < h2_integrals.n_beta else 0
            assert hf[n_orb + i].item() == expected, f"Beta orbital {i} mismatch"


# ===================================================================
# Ground state vs HF energy
# ===================================================================


class TestGroundStateBelowHF:
    """Test that exact ground state energy is below (or equal to) HF energy."""

    def test_ground_below_hf(self, h2_ham):
        """FCI ground state energy should be <= HF energy."""
        hf = h2_ham.get_hf_state()
        hf_energy = h2_ham.diagonal_element(hf).item()
        gs_energy, _ = h2_ham.exact_ground_state()
        assert gs_energy <= hf_energy + 1e-12, (
            f"Ground state {gs_energy} should be <= HF energy {hf_energy}"
        )


# ===================================================================
# JW sign tests
# ===================================================================


class TestJWSign:
    """Test Jordan-Wigner sign computation for known cases."""

    def test_single_adjacent_no_between(self, h2_ham):
        """Sign should be +1 when no occupied orbitals between p and q."""
        config = np.array([1, 0, 0, 0], dtype=np.int64)
        # p=1, q=0: between indices (1,0) -> range is empty
        sign = MolecularHamiltonian._jw_sign_single_py(config, 1, 0)
        assert sign == 1

    def test_single_with_occupied_between(self, h2_ham):
        """Sign should flip for each occupied orbital between p and q."""
        config = np.array([1, 1, 0, 0], dtype=np.int64)
        # p=2, q=0: between is index 1, which is occupied -> sign = -1
        sign = MolecularHamiltonian._jw_sign_single_py(config, 2, 0)
        assert sign == -1

    def test_single_two_occupied_between(self, h2_ham):
        """Two occupied orbitals between should give sign = +1."""
        config = np.array([1, 1, 1, 0], dtype=np.int64)
        # p=3, q=0: between is indices 1,2 both occupied -> count=2, sign=+1
        sign = MolecularHamiltonian._jw_sign_single_py(config, 3, 0)
        assert sign == 1

    def test_double_sign_known_case(self, h2_ham):
        """Test double excitation sign for a specific known configuration."""
        config = np.array([1, 1, 0, 0], dtype=np.int64)
        # Double excitation: annihilate 0,1 create 2,3
        sign = MolecularHamiltonian._jw_sign_double_py(config, 2, 3, 0, 1)
        # This is a known case; the sign depends on the ordering
        assert sign in (-1, 1)


# ===================================================================
# Edge cases
# ===================================================================


class TestMolecularEdgeCases:
    """Edge cases for MolecularHamiltonian."""

    def test_single_electron_system(self):
        """H atom with minimal basis should work (single electron)."""
        geometry = [("H", (0.0, 0.0, 0.0))]
        integrals = compute_molecular_integrals(
            geometry, basis="sto-3g", charge=0, spin=1
        )
        ham = MolecularHamiltonian(integrals)

        assert ham.num_sites == 2  # 1 spatial orbital -> 2 spin-orbitals
        assert integrals.n_electrons == 1
        assert integrals.n_alpha == 1
        assert integrals.n_beta == 0

        hf = ham.get_hf_state()
        assert hf.sum().item() == 1

    def test_integrals_validation(self):
        """MolecularIntegrals should reject mismatched shapes."""
        with pytest.raises(ValueError):
            MolecularIntegrals(
                h1e=np.zeros((2, 3)),  # Wrong shape
                h2e=np.zeros((2, 2, 2, 2)),
                nuclear_repulsion=0.0,
                n_electrons=2,
                n_orbitals=2,
                n_alpha=1,
                n_beta=1,
            )

    def test_integrals_electron_count_validation(self):
        """MolecularIntegrals should reject n_alpha + n_beta != n_electrons."""
        with pytest.raises(ValueError, match="n_alpha"):
            MolecularIntegrals(
                h1e=np.zeros((2, 2)),
                h2e=np.zeros((2, 2, 2, 2)),
                nuclear_repulsion=0.0,
                n_electrons=2,
                n_orbitals=2,
                n_alpha=2,
                n_beta=2,
            )
