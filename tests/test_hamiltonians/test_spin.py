"""Tests for spin Hamiltonians (Heisenberg and Transverse Field Ising)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qvartools.hamiltonians import HeisenbergHamiltonian, TransverseFieldIsing


# ===================================================================
# Heisenberg: diagonal element tests
# ===================================================================


class TestHeisenbergDiagonal:
    """Test Heisenberg diagonal_element for known configurations."""

    def test_all_up_periodic(self):
        """All spins up: S^z_i = +0.5 for all i, ZZ = +0.25 per bond."""
        h = HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True)
        config = torch.tensor([0, 0, 0, 0], dtype=torch.int64)  # all up
        diag = h.diagonal_element(config)
        # 4 bonds (periodic), each Jz * 0.5 * 0.5 = 0.25
        expected = 4 * 1.0 * 0.25
        assert diag.item() == pytest.approx(expected, abs=1e-12)

    def test_alternating_periodic(self):
        """Alternating up-down: each bond has S^z_i S^z_j = -0.25."""
        h = HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True)
        config = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        diag = h.diagonal_element(config)
        # 4 bonds, each Jz * (+0.5)*(-0.5) = -0.25
        expected = 4 * 1.0 * (-0.25)
        assert diag.item() == pytest.approx(expected, abs=1e-12)

    def test_all_down_periodic(self):
        """All spins down: same energy as all up (Jz * 0.25 per bond)."""
        h = HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True)
        config = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
        diag = h.diagonal_element(config)
        expected = 4 * 1.0 * 0.25
        assert diag.item() == pytest.approx(expected, abs=1e-12)

    def test_batch_matches_individual(self, heisenberg_4site):
        """Batch diagonal should match individual calls."""
        configs = heisenberg_4site._generate_all_configs()
        batch = heisenberg_4site.diagonal_elements_batch(configs)
        for idx in range(configs.shape[0]):
            individual = heisenberg_4site.diagonal_element(configs[idx])
            assert batch[idx].item() == pytest.approx(individual.item(), abs=1e-12)


# ===================================================================
# Heisenberg: get_connections tests
# ===================================================================


class TestHeisenbergConnections:
    """Test Heisenberg off-diagonal connections."""

    def test_all_up_no_flip_connections(self):
        """All-up state has no Sz-conserving pair flips for isotropic Heisenberg."""
        h = HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True)
        config = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        connected, elements = h.get_connections(config)
        # For isotropic Heisenberg (Jx=Jy), coeff_double = 0.25*(Jx-Jy) = 0
        # All aligned -> no S+S- flips possible since spins must differ
        # But S-S- terms: Jx != Jy => coeff_double = 0 for isotropic
        # All same spin -> no anti-aligned pair -> no S+S- either
        assert connected.shape[0] == 0

    def test_anti_aligned_pair_produces_flip(self):
        """An anti-aligned pair should produce a spin-flip connection."""
        h = HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True)
        config = torch.tensor([0, 1, 0, 0], dtype=torch.int64)  # site 1 is down
        connected, elements = h.get_connections(config)
        assert connected.shape[0] > 0

    def test_connections_are_valid_spin_flips(self, heisenberg_4site):
        """Connected states should differ from input by exactly 2 spin flips (pair exchange)."""
        config = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        connected, elements = heisenberg_4site.get_connections(config)
        for k in range(connected.shape[0]):
            diff = (connected[k] != config).sum().item()
            # Pair flip changes exactly 2 sites
            assert diff == 2, f"Connection {k} differs at {diff} sites (expected 2)"

    def test_connections_conserve_total_sz(self, heisenberg_4site):
        """For isotropic Heisenberg (Jx=Jy), total Sz is conserved."""
        config = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
        total_sz = config.sum().item()
        connected, _ = heisenberg_4site.get_connections(config)
        for k in range(connected.shape[0]):
            assert connected[k].sum().item() == total_sz


# ===================================================================
# Heisenberg: to_dense / Hermiticity tests
# ===================================================================


class TestHeisenbergDense:
    """Test dense matrix properties of Heisenberg Hamiltonian."""

    def test_is_hermitian(self, heisenberg_4site):
        """Dense matrix should be Hermitian (symmetric for real H)."""
        mat = heisenberg_4site.to_dense()
        assert torch.allclose(mat, mat.T, atol=1e-12)

    def test_trace(self):
        """Trace of isotropic Heisenberg should equal sum of ZZ diagonal."""
        h = HeisenbergHamiltonian(num_spins=2, Jx=1.0, Jy=1.0, Jz=1.0, periodic=False)
        mat = h.to_dense()
        # 2-site open chain: 1 bond
        # Diagonal elements: |00>: +0.25, |01>: -0.25, |10>: -0.25, |11>: +0.25
        # Trace = 0
        assert mat.trace().item() == pytest.approx(0.0, abs=1e-12)


# ===================================================================
# Heisenberg: exact ground state for 2-site
# ===================================================================


class TestHeisenberg2SiteExact:
    """Test exact ground state for 2-site Heisenberg (analytically known)."""

    def test_antiferro_2site_ground_energy(self):
        """2-site antiferromagnetic Heisenberg: E_gs = -0.75 * J for isotropic.

        H = J(S^x S^x + S^y S^y + S^z S^z)
        Singlet energy = J * (-3/4) = -0.75 for J=1.
        """
        h = HeisenbergHamiltonian(
            num_spins=2, Jx=1.0, Jy=1.0, Jz=1.0, periodic=False
        )
        energy, state = h.exact_ground_state()
        assert energy == pytest.approx(-0.75, abs=1e-10)

    def test_anisotropic_2site(self):
        """2-site anisotropic: ground state energy from exact diag."""
        h = HeisenbergHamiltonian(
            num_spins=2, Jx=1.0, Jy=1.0, Jz=2.0, periodic=False
        )
        energy, _ = h.exact_ground_state()
        # For XXZ with Jz=2, the ground state energy can be computed:
        # The 4x4 matrix for 2-site Heisenberg open chain:
        # |00>: Jz/4 = 0.5
        # |11>: Jz/4 = 0.5
        # |01>, |10> subspace: diag = -Jz/4 = -0.5, off-diag = (Jx+Jy)/4 = 0.5
        # Eigenvalues of 2x2: -0.5 +/- 0.5 -> 0 and -1
        # So ground state = -1
        assert energy == pytest.approx(-1.0, abs=1e-10)

    def test_ferro_2site(self):
        """2-site ferromagnetic (J < 0): triplet is ground state."""
        h = HeisenbergHamiltonian(
            num_spins=2, Jx=-1.0, Jy=-1.0, Jz=-1.0, periodic=False
        )
        energy, _ = h.exact_ground_state()
        # Ferro: triplet energy = J * 1/4 = -0.25 per bond
        # But we need to check: the Hamiltonian is J S.S
        # Triplet: <S.S> = 1/4, Energy = J * 1/4 = -1 * 1/4 = -0.25
        # Singlet: <S.S> = -3/4, Energy = J * (-3/4) = -1 * (-3/4) = 0.75
        # Ground state = -0.25
        assert energy == pytest.approx(-0.25, abs=1e-10)


# ===================================================================
# TFIM: diagonal element tests
# ===================================================================


class TestTFIMDiagonal:
    """Test TFIM diagonal elements."""

    def test_all_up(self):
        """All up: each ZZ pair gives -V * 0.25."""
        h = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0, periodic=True)
        config = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        diag = h.diagonal_element(config)
        # 4 bonds, each -V * (0.5)(0.5) = -0.25
        expected = 4 * (-1.0 * 0.25)
        assert diag.item() == pytest.approx(expected, abs=1e-12)

    def test_alternating(self):
        """Alternating: each ZZ pair gives -V * (-0.25) = +V*0.25."""
        h = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0, periodic=True)
        config = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        diag = h.diagonal_element(config)
        expected = 4 * (-1.0 * (-0.25))
        assert diag.item() == pytest.approx(expected, abs=1e-12)

    def test_zero_field_diagonal(self):
        """With h=0, diagonal elements should still be correct."""
        h = TransverseFieldIsing(num_spins=3, V=2.0, h=0.0, periodic=False)
        config = torch.tensor([0, 0, 0], dtype=torch.int64)
        diag = h.diagonal_element(config)
        # 2 bonds (open), each -2.0 * 0.25 = -0.5
        expected = 2 * (-2.0 * 0.25)
        assert diag.item() == pytest.approx(expected, abs=1e-12)


# ===================================================================
# TFIM: off-diagonal element tests
# ===================================================================


class TestTFIMConnections:
    """Test TFIM off-diagonal connections."""

    def test_each_spin_flipped(self):
        """Transverse field should produce N single-spin-flip connections."""
        h = TransverseFieldIsing(num_spins=4, V=1.0, h=1.0, periodic=True)
        config = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        connected, elements = h.get_connections(config)
        assert connected.shape[0] == 4

    def test_flip_amplitude(self):
        """Each flip should have amplitude -h/2."""
        h = TransverseFieldIsing(num_spins=3, V=1.0, h=2.0, periodic=True)
        config = torch.tensor([0, 0, 0], dtype=torch.int64)
        _, elements = h.get_connections(config)
        expected_coeff = -0.5 * 2.0  # -h/2
        for k in range(elements.shape[0]):
            assert elements[k].item() == pytest.approx(expected_coeff, abs=1e-12)

    def test_no_connections_when_h_zero(self):
        """With h=0, there should be no off-diagonal connections."""
        h = TransverseFieldIsing(num_spins=4, V=1.0, h=0.0, periodic=True)
        config = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        connected, _ = h.get_connections(config)
        assert connected.shape[0] == 0

    def test_connections_differ_by_one_spin(self, tfim_6site):
        """Each connection should differ from config at exactly 1 site."""
        config = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int64)
        connected, _ = tfim_6site.get_connections(config)
        for k in range(connected.shape[0]):
            diff = (connected[k] != config).sum().item()
            assert diff == 1


# ===================================================================
# TFIM: ground state tests
# ===================================================================


class TestTFIMGroundState:
    """Test TFIM ground state properties."""

    def test_ground_state_below_all_diagonals(self, tfim_6site):
        """Ground state energy should be below the minimum diagonal element."""
        gs_energy, _ = tfim_6site.exact_ground_state()
        configs = tfim_6site._generate_all_configs()
        diag = torch.stack(
            [tfim_6site.diagonal_element(configs[i]) for i in range(configs.shape[0])]
        )
        min_diag = diag.min().item()
        assert gs_energy < min_diag + 1e-10, (
            f"Ground state {gs_energy} should be below min diagonal {min_diag}"
        )

    def test_ground_state_is_eigenstate(self, tfim_6site):
        """H|psi_gs> should equal E_gs * |psi_gs>."""
        gs_energy, gs_state = tfim_6site.exact_ground_state()
        mat = tfim_6site.to_dense()
        h_psi = mat @ gs_state
        e_psi = gs_energy * gs_state
        assert torch.allclose(h_psi, e_psi, atol=1e-10)

    def test_small_system_known_limit(self):
        """2-site TFIM at h=0: ground state energy should be -V/4."""
        h = TransverseFieldIsing(num_spins=2, V=4.0, h=0.0, periodic=False)
        energy, _ = h.exact_ground_state()
        # 1 bond, -V * S^z_0 S^z_1, minimum at aligned spins: -4 * 0.25 = -1
        assert energy == pytest.approx(-1.0, abs=1e-10)


# ===================================================================
# Periodic vs open boundary conditions
# ===================================================================


class TestBoundaryConditions:
    """Test periodic vs open boundary conditions."""

    def test_heisenberg_periodic_vs_open_differ(self):
        """Periodic and open Heisenberg should give different energies for N>2."""
        h_periodic = HeisenbergHamiltonian(
            num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True
        )
        h_open = HeisenbergHamiltonian(
            num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=False
        )
        e_periodic, _ = h_periodic.exact_ground_state()
        e_open, _ = h_open.exact_ground_state()
        assert e_periodic != pytest.approx(e_open, abs=1e-6), (
            "Periodic and open 4-site Heisenberg should have different ground energies"
        )

    def test_tfim_periodic_vs_open_differ(self):
        """Periodic and open TFIM should give different energies for N>2."""
        h_periodic = TransverseFieldIsing(
            num_spins=4, V=1.0, h=1.0, periodic=True
        )
        h_open = TransverseFieldIsing(
            num_spins=4, V=1.0, h=1.0, periodic=False
        )
        e_periodic, _ = h_periodic.exact_ground_state()
        e_open, _ = h_open.exact_ground_state()
        assert e_periodic != pytest.approx(e_open, abs=1e-6)

    def test_heisenberg_open_fewer_bonds(self):
        """Open chain should have fewer bonds than periodic."""
        h_open = HeisenbergHamiltonian(
            num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=False
        )
        h_periodic = HeisenbergHamiltonian(
            num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True
        )
        assert len(h_open._neighbours) == 3
        assert len(h_periodic._neighbours) == 4

    def test_tfim_open_fewer_interactions(self):
        """Open TFIM should have fewer interaction pairs."""
        h_open = TransverseFieldIsing(
            num_spins=4, V=1.0, h=1.0, periodic=False
        )
        h_periodic = TransverseFieldIsing(
            num_spins=4, V=1.0, h=1.0, periodic=True
        )
        assert len(h_open._interaction_pairs) == 3
        assert len(h_periodic._interaction_pairs) == 4

    def test_2site_periodic_equals_open(self):
        """For 2 sites, periodic and open Heisenberg should give same energy.

        Periodic would connect site 0->1 and 1->0 (same bond), but
        actually for 2 sites periodic gives 2 bonds while open gives 1.
        So they differ even for N=2.
        """
        h_periodic = HeisenbergHamiltonian(
            num_spins=2, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True
        )
        h_open = HeisenbergHamiltonian(
            num_spins=2, Jx=1.0, Jy=1.0, Jz=1.0, periodic=False
        )
        # Periodic 2-site has bonds (0,1) and (1,0) -- 2 bonds counting the wrap
        # Open 2-site has only bond (0,1) -- 1 bond
        assert len(h_periodic._neighbours) == 2
        assert len(h_open._neighbours) == 1


# ===================================================================
# Heisenberg: dense matrix is Hermitian (using fixture)
# ===================================================================


class TestHeisenbergDenseFixture:
    """Additional tests using the fixture."""

    def test_dense_hermitian(self, heisenberg_4site):
        mat = heisenberg_4site.to_dense()
        assert torch.allclose(mat, mat.T, atol=1e-12)

    def test_sparse_matches_dense(self, heisenberg_4site):
        dense = heisenberg_4site.to_dense().numpy()
        sparse = heisenberg_4site.to_sparse().toarray()
        np.testing.assert_allclose(sparse, dense, atol=1e-12)


class TestTFIMDenseFixture:
    """Additional tests using the TFIM fixture."""

    def test_dense_hermitian(self, tfim_6site):
        mat = tfim_6site.to_dense()
        assert torch.allclose(mat, mat.T, atol=1e-12)

    def test_sparse_matches_dense(self, tfim_6site):
        dense = tfim_6site.to_dense().numpy()
        sparse = tfim_6site.to_sparse().toarray()
        np.testing.assert_allclose(sparse, dense, atol=1e-12)
