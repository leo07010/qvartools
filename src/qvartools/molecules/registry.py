"""
registry --- Molecule registry and factory functions
====================================================

Defines a registry of standard molecular benchmarks used in quantum
chemistry.  Each entry provides a factory function that computes
molecular integrals and constructs a :class:`MolecularHamiltonian`.

The registry covers a range of system sizes from H2 (4 qubits) to
C2H4 (28 qubits), enabling systematic benchmarking of SQD/SKQD methods.

Constants
---------
MOLECULE_REGISTRY
    Dictionary mapping lowercase molecule names to factory metadata.

Functions
---------
get_molecule
    Instantiate a Hamiltonian for a named molecule.
list_molecules
    Return sorted list of registered molecule names.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from qvartools.hamiltonians.molecular import (
    MolecularHamiltonian,
    compute_molecular_integrals,
)

__all__ = [
    "MOLECULE_REGISTRY",
    "get_molecule",
    "list_molecules",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry definitions
# ---------------------------------------------------------------------------

_H2_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("H", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 0.74)),
]

_LIH_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("Li", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.6)),
]

_BEH2_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("Be", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 1.33)),
    ("H", (0.0, 0.0, -1.33)),
]

# H2O geometry from parametric (OH=0.96 Å, angle=104.5°)
import math as _math

_H2O_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.96, 0.0, 0.0)),
    ("H", (0.96 * _math.cos(_math.radians(104.5)),
           0.96 * _math.sin(_math.radians(104.5)), 0.0)),
]

_NH3_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("N", (0.0, 0.0, 0.0)),
    ("H", (0.0, -0.9377, -0.3816)),
    ("H", (0.8121, 0.4689, -0.3816)),
    ("H", (-0.8121, 0.4689, -0.3816)),
]

_N2_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("N", (0.0, 0.0, 0.0)),
    ("N", (0.0, 0.0, 1.0977)),
]

_CH4_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("C", (0.0, 0.0, 0.0)),
    ("H", (0.6276, 0.6276, 0.6276)),
    ("H", (0.6276, -0.6276, -0.6276)),
    ("H", (-0.6276, 0.6276, -0.6276)),
    ("H", (-0.6276, -0.6276, 0.6276)),
]

_C2H4_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("C", (0.0, 0.0, 0.6695)),
    ("C", (0.0, 0.0, -0.6695)),
    ("H", (0.0, 0.9289, 1.2321)),
    ("H", (0.0, -0.9289, 1.2321)),
    ("H", (0.0, 0.9289, -1.2321)),
    ("H", (0.0, -0.9289, -1.2321)),
]

_CO_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.0, 0.0, 1.13)),
]

_HCN_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("H", (0.0, 0.0, 0.0)),
    ("C", (0.0, 0.0, 1.06)),
    ("N", (0.0, 0.0, 2.22)),
]

_C2H2_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("H", (0.0, 0.0, 0.0)),
    ("C", (0.0, 0.0, 1.06)),
    ("C", (0.0, 0.0, 2.26)),
    ("H", (0.0, 0.0, 3.32)),
]

_H2S_GEOMETRY: List[Tuple[str, Tuple[float, float, float]]] = [
    ("S", (0.0, 0.0, 0.0)),
    ("H", (1.34, 0.0, 0.0)),
    ("H", (-0.0497, 1.3391, 0.0)),
]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _make_h2(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create H2 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_H2_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("H2", 4, "sto-3g", _H2_GEOMETRY, 0, 0)
    return ham, info


def _make_lih(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create LiH Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_LIH_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("LiH", 12, "sto-3g", _LIH_GEOMETRY, 0, 0)
    return ham, info


def _make_beh2(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create BeH2 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_BEH2_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("BeH2", 14, "sto-3g", _BEH2_GEOMETRY, 0, 0)
    return ham, info


def _make_h2o(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create H2O Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_H2O_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("H2O", 14, "sto-3g", _H2O_GEOMETRY, 0, 0)
    return ham, info


def _make_nh3(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create NH3 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_NH3_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("NH3", 16, "sto-3g", _NH3_GEOMETRY, 0, 0)
    return ham, info


def _make_n2(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create N2 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_N2_GEOMETRY, basis="cc-pvdz", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("N2", 20, "cc-pvdz", _N2_GEOMETRY, 0, 0)
    return ham, info


def _make_ch4(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create CH4 Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_CH4_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("CH4", 18, "sto-3g", _CH4_GEOMETRY, 0, 0)
    return ham, info


def _make_c2h4(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create C2H4 (ethylene) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_C2H4_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("C2H4", 28, "sto-3g", _C2H4_GEOMETRY, 0, 0)
    return ham, info


def _make_co(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create CO (carbon monoxide) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_CO_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("CO", 20, "sto-3g", _CO_GEOMETRY, 0, 0)
    return ham, info


def _make_hcn(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create HCN (hydrogen cyanide) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_HCN_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("HCN", 22, "sto-3g", _HCN_GEOMETRY, 0, 0)
    return ham, info


def _make_c2h2(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create C2H2 (acetylene) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_C2H2_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("C2H2", 24, "sto-3g", _C2H2_GEOMETRY, 0, 0)
    return ham, info


def _make_h2s(device: str = "cpu") -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create H2S (hydrogen sulfide) Hamiltonian and info dict.

    Parameters
    ----------
    device : str, optional
        Torch device (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(hamiltonian, info_dict)``.
    """
    integrals = compute_molecular_integrals(
        geometry=_H2S_GEOMETRY, basis="sto-3g", charge=0, spin=0
    )
    ham = MolecularHamiltonian(integrals, device=device)
    info = _build_info("H2S", 26, "sto-3g", _H2S_GEOMETRY, 0, 0)
    return ham, info


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_info(
    name: str,
    n_qubits: int,
    basis: str,
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    charge: int,
    spin: int,
) -> Dict[str, Any]:
    """Build a standardised molecule info dictionary.

    Parameters
    ----------
    name : str
        Molecule name.
    n_qubits : int
        Number of qubits (spin-orbitals).
    basis : str
        Gaussian basis set.
    geometry : list
        Atomic geometry.
    charge : int
        Net molecular charge.
    spin : int
        Spin multiplicity minus one (2S).

    Returns
    -------
    dict
        Info dictionary with keys ``name``, ``n_qubits``, ``basis``,
        ``geometry``, ``charge``, ``spin``.
    """
    return {
        "name": name,
        "n_qubits": n_qubits,
        "basis": basis,
        "geometry": geometry,
        "charge": charge,
        "spin": spin,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MOLECULE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "h2": {
        "factory": _make_h2,
        "n_qubits": 4,
        "description": "Hydrogen molecule (minimal basis)",
        "basis": "sto-3g",
    },
    "lih": {
        "factory": _make_lih,
        "n_qubits": 12,
        "description": "Lithium hydride",
        "basis": "sto-3g",
    },
    "beh2": {
        "factory": _make_beh2,
        "n_qubits": 14,
        "description": "Beryllium dihydride",
        "basis": "sto-3g",
    },
    "h2o": {
        "factory": _make_h2o,
        "n_qubits": 14,
        "description": "Water molecule",
        "basis": "sto-3g",
    },
    "nh3": {
        "factory": _make_nh3,
        "n_qubits": 16,
        "description": "Ammonia",
        "basis": "sto-3g",
    },
    "n2": {
        "factory": _make_n2,
        "n_qubits": 20,
        "description": "Nitrogen molecule (cc-pVDZ basis)",
        "basis": "cc-pvdz",
    },
    "ch4": {
        "factory": _make_ch4,
        "n_qubits": 18,
        "description": "Methane",
        "basis": "sto-3g",
    },
    "c2h4": {
        "factory": _make_c2h4,
        "n_qubits": 28,
        "description": "Ethylene (minimal basis)",
        "basis": "sto-3g",
    },
    "co": {
        "factory": _make_co,
        "n_qubits": 20,
        "description": "Carbon monoxide (STO-3G)",
        "basis": "sto-3g",
    },
    "hcn": {
        "factory": _make_hcn,
        "n_qubits": 22,
        "description": "Hydrogen cyanide (STO-3G)",
        "basis": "sto-3g",
    },
    "c2h2": {
        "factory": _make_c2h2,
        "n_qubits": 24,
        "description": "Acetylene (STO-3G)",
        "basis": "sto-3g",
    },
    "h2s": {
        "factory": _make_h2s,
        "n_qubits": 26,
        "description": "Hydrogen sulfide (STO-3G)",
        "basis": "sto-3g",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_molecule(
    name: str, device: str = "cpu"
) -> Tuple[MolecularHamiltonian, Dict[str, Any]]:
    """Create a Hamiltonian and info dict for a named molecule.

    Looks up the molecule in :data:`MOLECULE_REGISTRY`, runs the PySCF
    integral computation, and constructs a :class:`MolecularHamiltonian`.

    Parameters
    ----------
    name : str
        Molecule name (case-insensitive).  Must be a key in
        :data:`MOLECULE_REGISTRY`.
    device : str, optional
        Torch device for the Hamiltonian (default ``"cpu"``).

    Returns
    -------
    hamiltonian : MolecularHamiltonian
        The molecular Hamiltonian ready for diagonalisation.
    info : dict
        Metadata dictionary with keys ``name``, ``n_qubits``, ``basis``,
        ``geometry``, ``charge``, ``spin``.

    Raises
    ------
    KeyError
        If ``name`` is not found in the registry.

    Examples
    --------
    >>> ham, info = get_molecule("H2")
    >>> info["n_qubits"]
    4
    >>> ham.num_sites
    4
    """
    key = name.lower().strip()
    if key not in MOLECULE_REGISTRY:
        available = ", ".join(sorted(MOLECULE_REGISTRY.keys()))
        raise KeyError(
            f"Unknown molecule {name!r}. Available: {available}"
        )

    entry = MOLECULE_REGISTRY[key]
    factory = entry["factory"]

    logger.info(
        "Creating molecule %r (%d qubits, %s basis)",
        key,
        entry["n_qubits"],
        entry["basis"],
    )

    return factory(device=device)


def list_molecules() -> List[str]:
    """Return a sorted list of available molecule names.

    Returns
    -------
    list of str
        Registered molecule names in alphabetical order.

    Examples
    --------
    >>> names = list_molecules()
    >>> "h2" in names
    True
    """
    return sorted(MOLECULE_REGISTRY.keys())
