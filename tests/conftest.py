"""Shared pytest fixtures for qvartools test suite."""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# PySCF availability
# ---------------------------------------------------------------------------

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = True
except ImportError:
    _HAS_PYSCF = False

pyscf_required = pytest.mark.skipif(
    not _HAS_PYSCF, reason="PySCF is not installed"
)


# ---------------------------------------------------------------------------
# Molecular fixtures (require PySCF)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def h2_hamiltonian():
    """H2 molecule at 0.74 angstrom, sto-3g basis (4 qubits)."""
    if not _HAS_PYSCF:
        pytest.skip("PySCF is not installed")
    from qvartools.hamiltonians import (
        MolecularHamiltonian,
        compute_molecular_integrals,
    )

    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return MolecularHamiltonian(integrals)


@pytest.fixture(scope="session")
def beh2_hamiltonian():
    """BeH2 molecule, sto-6g basis (14 qubits)."""
    if not _HAS_PYSCF:
        pytest.skip("PySCF is not installed")
    from qvartools.hamiltonians import (
        MolecularHamiltonian,
        compute_molecular_integrals,
    )

    geometry = [
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.3264)),
        ("H", (0.0, 0.0, -1.3264)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-6g")
    return MolecularHamiltonian(integrals)


# ---------------------------------------------------------------------------
# Spin-model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def heisenberg_4site():
    """4-site Heisenberg model (Jx=Jy=Jz=1.0, periodic)."""
    from qvartools.hamiltonians import HeisenbergHamiltonian

    return HeisenbergHamiltonian(num_spins=4, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True)


@pytest.fixture(scope="session")
def tfim_6site():
    """6-site Transverse Field Ising model (V=1.0, h=1.0, periodic)."""
    from qvartools.hamiltonians import TransverseFieldIsing

    return TransverseFieldIsing(num_spins=6, V=1.0, h=1.0, periodic=True)
