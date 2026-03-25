"""Tests for Hamiltonian base class and PauliString."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian, PauliString


# ---------------------------------------------------------------------------
# Concrete Hamiltonian for testing the ABC helpers
# ---------------------------------------------------------------------------


class _SimpleZZHamiltonian(Hamiltonian):
    """Minimal 2-qubit ZZ Hamiltonian for testing base class methods.

    H = Z_0 Z_1  (diagonal only, no off-diagonal connections).
    Eigenvalues: +1 for |00>, |11>; -1 for |01>, |10>.
    """

    def __init__(self) -> None:
        super().__init__(num_sites=2, local_dim=2)

    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        # Z_0 Z_1: (+1)(+1) = +1 for 00, (+1)(-1) = -1 for 01, etc.
        s0 = 1.0 - 2.0 * float(config[0].item())
        s1 = 1.0 - 2.0 * float(config[1].item())
        return torch.tensor(s0 * s1, dtype=torch.float64)

    def get_connections(self, config: torch.Tensor):
        # Purely diagonal -- no off-diagonal connections
        return (
            torch.empty((0, self.num_sites), dtype=torch.int64),
            torch.empty(0, dtype=torch.float64),
        )


class _SimpleXXHamiltonian(Hamiltonian):
    """Minimal 2-qubit XX Hamiltonian for testing off-diagonal paths.

    H = X_0 X_1.  Off-diagonal: flips both qubits.
    """

    def __init__(self) -> None:
        super().__init__(num_sites=2, local_dim=2)

    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, dtype=torch.float64)

    def get_connections(self, config: torch.Tensor):
        flipped = 1 - config
        return (
            flipped.unsqueeze(0).to(torch.int64),
            torch.tensor([1.0], dtype=torch.float64),
        )


# ===================================================================
# PauliString.apply tests
# ===================================================================


class TestPauliStringApply:
    """Test PauliString.apply with individual Pauli operators."""

    def test_identity_on_zero(self):
        ps = PauliString(["I"])
        new_cfg, phase = ps.apply(torch.tensor([0]))
        assert new_cfg.tolist() == [0]
        assert phase == 1.0 + 0j

    def test_identity_on_one(self):
        ps = PauliString(["I"])
        new_cfg, phase = ps.apply(torch.tensor([1]))
        assert new_cfg.tolist() == [1]
        assert phase == 1.0 + 0j

    def test_x_on_zero(self):
        ps = PauliString(["X"])
        new_cfg, phase = ps.apply(torch.tensor([0]))
        assert new_cfg.tolist() == [1]
        assert phase == 1.0 + 0j

    def test_x_on_one(self):
        ps = PauliString(["X"])
        new_cfg, phase = ps.apply(torch.tensor([1]))
        assert new_cfg.tolist() == [0]
        assert phase == 1.0 + 0j

    def test_y_on_zero(self):
        ps = PauliString(["Y"])
        new_cfg, phase = ps.apply(torch.tensor([0]))
        assert new_cfg.tolist() == [1]
        assert phase == pytest.approx(1j)

    def test_y_on_one(self):
        ps = PauliString(["Y"])
        new_cfg, phase = ps.apply(torch.tensor([1]))
        assert new_cfg.tolist() == [0]
        assert phase == pytest.approx(-1j)

    def test_z_on_zero(self):
        ps = PauliString(["Z"])
        new_cfg, phase = ps.apply(torch.tensor([0]))
        assert new_cfg.tolist() == [0]
        assert phase == 1.0 + 0j

    def test_z_on_one(self):
        ps = PauliString(["Z"])
        new_cfg, phase = ps.apply(torch.tensor([1]))
        assert new_cfg.tolist() == [1]
        assert phase == -1.0 + 0j

    def test_coefficient_propagates(self):
        ps = PauliString(["X"], coefficient=0.5)
        _, phase = ps.apply(torch.tensor([0]))
        assert phase == pytest.approx(0.5)

    def test_multi_qubit(self):
        ps = PauliString(["X", "Z", "I"])
        new_cfg, phase = ps.apply(torch.tensor([0, 1, 0]))
        assert new_cfg.tolist() == [1, 1, 0]
        assert phase == pytest.approx(-1.0)  # X gives 1, Z on |1> gives -1, I gives 1


# ===================================================================
# PauliString.is_diagonal tests
# ===================================================================


class TestPauliStringIsDiagonal:
    """Test PauliString.is_diagonal detection."""

    def test_all_identity(self):
        assert PauliString(["I", "I", "I"]).is_diagonal() is True

    def test_all_z(self):
        assert PauliString(["Z", "Z"]).is_diagonal() is True

    def test_iz_mix(self):
        assert PauliString(["I", "Z", "I", "Z"]).is_diagonal() is True

    def test_contains_x(self):
        assert PauliString(["I", "X"]).is_diagonal() is False

    def test_contains_y(self):
        assert PauliString(["Y", "Z"]).is_diagonal() is False

    def test_single_x(self):
        assert PauliString(["X"]).is_diagonal() is False


# ===================================================================
# _config_to_index / _index_to_config round-trip
# ===================================================================


class TestConfigIndexRoundTrip:
    """Test that _config_to_index and _index_to_config are inverses."""

    def test_round_trip_all_configs_2site(self):
        h = _SimpleZZHamiltonian()
        for idx in range(h.hilbert_dim):
            config = h._index_to_config(idx)
            recovered_idx = h._config_to_index(config)
            assert recovered_idx == idx

    def test_specific_configs(self):
        h = _SimpleZZHamiltonian()
        # |00> -> index 0, |01> -> 1, |10> -> 2, |11> -> 3
        assert h._config_to_index(torch.tensor([0, 0])) == 0
        assert h._config_to_index(torch.tensor([0, 1])) == 1
        assert h._config_to_index(torch.tensor([1, 0])) == 2
        assert h._config_to_index(torch.tensor([1, 1])) == 3

    def test_round_trip_3site(self):
        """Test with 3 sites to cover non-trivial local_dim powers."""

        class _Trivial3(Hamiltonian):
            def __init__(self):
                super().__init__(num_sites=3, local_dim=2)

            def diagonal_element(self, config):
                return torch.tensor(0.0, dtype=torch.float64)

            def get_connections(self, config):
                return (
                    torch.empty((0, 3), dtype=torch.int64),
                    torch.empty(0, dtype=torch.float64),
                )

        h = _Trivial3()
        for idx in range(h.hilbert_dim):
            config = h._index_to_config(idx)
            assert h._config_to_index(config) == idx


# ===================================================================
# to_dense tests
# ===================================================================


class TestToDense:
    """Test dense matrix construction from the base class."""

    def test_zz_dense_is_symmetric(self):
        h = _SimpleZZHamiltonian()
        mat = h.to_dense()
        assert torch.allclose(mat, mat.T)

    def test_zz_dense_values(self):
        h = _SimpleZZHamiltonian()
        mat = h.to_dense()
        # Diagonal should be [+1, -1, -1, +1]
        expected_diag = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64)
        assert torch.allclose(mat.diag(), expected_diag)
        # Off-diagonal should be zero for ZZ
        off_diag = mat - torch.diag(mat.diag())
        assert torch.allclose(off_diag, torch.zeros_like(off_diag))

    def test_xx_dense_is_symmetric(self):
        h = _SimpleXXHamiltonian()
        mat = h.to_dense()
        assert torch.allclose(mat, mat.T)

    def test_xx_dense_off_diagonal(self):
        h = _SimpleXXHamiltonian()
        mat = h.to_dense()
        # X_0 X_1 flips both bits: |00> <-> |11>, |01> <-> |10>
        assert mat[0, 3].item() == pytest.approx(1.0)
        assert mat[3, 0].item() == pytest.approx(1.0)
        assert mat[1, 2].item() == pytest.approx(1.0)
        assert mat[2, 1].item() == pytest.approx(1.0)


# ===================================================================
# to_sparse tests
# ===================================================================


class TestToSparse:
    """Test sparse matrix matches dense matrix."""

    def test_sparse_matches_dense_zz(self):
        h = _SimpleZZHamiltonian()
        dense = h.to_dense().numpy()
        sparse = h.to_sparse().toarray()
        np.testing.assert_allclose(sparse, dense, atol=1e-12)

    def test_sparse_matches_dense_xx(self):
        h = _SimpleXXHamiltonian()
        dense = h.to_dense().numpy()
        sparse = h.to_sparse().toarray()
        np.testing.assert_allclose(sparse, dense, atol=1e-12)


# ===================================================================
# exact_ground_state tests
# ===================================================================


class TestExactGroundState:
    """Test exact_ground_state via dense diagonalisation."""

    def test_zz_ground_energy(self):
        h = _SimpleZZHamiltonian()
        energy, state = h.exact_ground_state()
        # Ground energy of Z_0 Z_1 is -1 (degenerate: |01>, |10>)
        assert energy == pytest.approx(-1.0, abs=1e-10)

    def test_xx_ground_energy(self):
        h = _SimpleXXHamiltonian()
        energy, state = h.exact_ground_state()
        # X_0 X_1 eigenvalues are +1, +1, -1, -1. Ground = -1.
        assert energy == pytest.approx(-1.0, abs=1e-10)

    def test_ground_state_normalized(self):
        h = _SimpleXXHamiltonian()
        _, state = h.exact_ground_state()
        norm = torch.linalg.norm(state)
        assert norm.item() == pytest.approx(1.0, abs=1e-10)

    def test_ground_state_matches_sparse(self):
        """Sparse matrix diag should agree with dense diagonalisation."""
        h = _SimpleXXHamiltonian()
        energy_dense, _ = h.exact_ground_state()
        # Use scipy dense eigh on the sparse matrix instead of eigsh
        # (eigsh fails on very small matrices where k is close to dim)
        sparse_mat = h.to_sparse()
        eigenvalues = np.linalg.eigvalsh(sparse_mat.toarray())
        assert eigenvalues[0] == pytest.approx(energy_dense, abs=1e-10)


# ===================================================================
# Edge cases / validation
# ===================================================================


class TestValidation:
    """Test input validation in the base class."""

    def test_invalid_num_sites(self):
        with pytest.raises(ValueError, match="num_sites must be >= 1"):
            _SimpleZZHamiltonian.__bases__[0].__init__(
                _SimpleZZHamiltonian.__new__(_SimpleZZHamiltonian), num_sites=0
            )

    def test_invalid_pauli_label(self):
        with pytest.raises(ValueError, match="Invalid Pauli label"):
            PauliString(["A"])
