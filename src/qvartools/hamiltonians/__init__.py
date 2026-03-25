"""
hamiltonians — Hamiltonian representations for quantum systems
==============================================================

This subpackage provides Hamiltonian classes for molecular and spin systems
with efficient matrix-element computation, diagonal/off-diagonal decomposition,
and exact diagonalisation utilities.

Classes
-------
Hamiltonian
    Abstract base class for all Hamiltonians.
PauliString
    Single Pauli string (tensor product of I, X, Y, Z) with a coefficient.
MolecularHamiltonian
    Second-quantised molecular Hamiltonian via Jordan--Wigner mapping.
MolecularIntegrals
    Container for one- and two-electron integrals plus metadata.
HeisenbergHamiltonian
    Anisotropic Heisenberg (XXZ/XYZ) model on a 1-D chain.
TransverseFieldIsing
    Transverse-field Ising model with tuneable interaction range.

Functions
---------
compute_molecular_integrals
    Run RHF via PySCF and return ``MolecularIntegrals``.
"""

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.hamiltonians.spin.heisenberg import HeisenbergHamiltonian
from qvartools.hamiltonians.integrals import (
    MolecularIntegrals,
    compute_molecular_integrals,
)
from qvartools.hamiltonians.molecular import MolecularHamiltonian
from qvartools.hamiltonians.pauli_string import PauliString
from qvartools.hamiltonians.spin.tfim import TransverseFieldIsing

__all__ = [
    "Hamiltonian",
    "PauliString",
    "MolecularHamiltonian",
    "MolecularIntegrals",
    "compute_molecular_integrals",
    "HeisenbergHamiltonian",
    "TransverseFieldIsing",
]
