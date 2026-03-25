"""
molecules --- Molecular system registry and factory functions
=============================================================

This subpackage provides a registry of standard molecular benchmarks and
factory functions for creating Hamiltonians and metadata dictionaries
ready for use with the solvers.

Constants
---------
MOLECULE_REGISTRY
    Dictionary mapping molecule names to factory information.

Functions
---------
get_molecule
    Create a Hamiltonian and info dict for a named molecule.
list_molecules
    Return sorted list of available molecule names.
"""

from qvartools.molecules.registry import (
    MOLECULE_REGISTRY,
    get_molecule,
    list_molecules,
)

__all__ = [
    "MOLECULE_REGISTRY",
    "get_molecule",
    "list_molecules",
]
