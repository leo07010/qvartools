"""
integrals — Molecular integral container and PySCF computation
==============================================================

Provides the ``MolecularIntegrals`` frozen dataclass that holds one- and
two-electron integrals together with molecule metadata, and the
``compute_molecular_integrals`` helper that runs RHF via PySCF and
returns a populated ``MolecularIntegrals`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = [
    "MATRIX_ELEMENT_TOL",
    "MolecularIntegrals",
    "compute_molecular_integrals",
]

MATRIX_ELEMENT_TOL: float = 1e-12
"""float : Absolute tolerance below which matrix elements are treated as zero."""


# ---------------------------------------------------------------------------
# MolecularIntegrals dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MolecularIntegrals:
    """Container for molecular one- and two-electron integrals.

    All arrays use the *spatial-orbital* indexing convention produced by
    PySCF's ``ao2mo`` module.

    Parameters
    ----------
    h1e : np.ndarray
        One-electron integrals, shape ``(n_orb, n_orb)``, dtype ``float64``.
    h2e : np.ndarray
        Two-electron integrals in chemist's notation ``(pq|rs)``,
        shape ``(n_orb, n_orb, n_orb, n_orb)``, dtype ``float64``.
    nuclear_repulsion : float
        Nuclear repulsion energy in Hartree.
    n_electrons : int
        Total number of electrons.
    n_orbitals : int
        Number of spatial orbitals.
    n_alpha : int
        Number of alpha (spin-up) electrons.
    n_beta : int
        Number of beta (spin-down) electrons.

    Raises
    ------
    ValueError
        If array shapes are inconsistent with ``n_orbitals`` or if
        ``n_alpha + n_beta != n_electrons``.

    Examples
    --------
    >>> import numpy as np
    >>> h1 = np.zeros((2, 2))
    >>> h2 = np.zeros((2, 2, 2, 2))
    >>> mi = MolecularIntegrals(h1, h2, 0.7, 2, 2, 1, 1)
    >>> mi.n_orbitals
    2
    """

    h1e: np.ndarray
    h2e: np.ndarray
    nuclear_repulsion: float
    n_electrons: int
    n_orbitals: int
    n_alpha: int
    n_beta: int

    def __post_init__(self) -> None:
        """Validate shapes and dtypes."""
        n = self.n_orbitals
        if self.h1e.shape != (n, n):
            raise ValueError(
                f"h1e shape {self.h1e.shape} does not match n_orbitals={n}"
            )
        if self.h2e.shape != (n, n, n, n):
            raise ValueError(
                f"h2e shape {self.h2e.shape} does not match n_orbitals={n}"
            )
        if self.n_alpha + self.n_beta != self.n_electrons:
            raise ValueError(
                f"n_alpha ({self.n_alpha}) + n_beta ({self.n_beta}) "
                f"!= n_electrons ({self.n_electrons})"
            )


# ---------------------------------------------------------------------------
# compute_molecular_integrals (PySCF)
# ---------------------------------------------------------------------------


def compute_molecular_integrals(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
) -> MolecularIntegrals:
    """Run RHF with PySCF and extract molecular integrals.

    Parameters
    ----------
    geometry : list of (str, (float, float, float))
        Molecular geometry.  Each element is ``(atom_symbol, (x, y, z))``
        with coordinates in **Angstroms**.
    basis : str, optional
        Gaussian basis set name (default ``"sto-3g"``).
    charge : int, optional
        Net charge of the molecule (default ``0``).
    spin : int, optional
        Spin multiplicity minus one, i.e. ``2S`` (default ``0`` for
        singlet).

    Returns
    -------
    MolecularIntegrals
        Integrals and metadata needed by :class:`MolecularHamiltonian`.

    Raises
    ------
    ImportError
        If PySCF is not installed.
    RuntimeError
        If the SCF calculation does not converge.

    Examples
    --------
    >>> geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
    >>> mi = compute_molecular_integrals(geometry, basis="sto-3g")  # doctest: +SKIP
    >>> mi.n_orbitals  # doctest: +SKIP
    2
    """
    try:
        import pyscf  # noqa: F401
        from pyscf import ao2mo, gto, scf
    except ImportError as exc:
        raise ImportError(
            "PySCF is required for compute_molecular_integrals. "
            "Install it with: pip install pyscf"
        ) from exc

    # Build molecule
    mol = gto.Mole()
    mol.atom = [(atom, coord) for atom, coord in geometry]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.unit = "Angstrom"
    mol.build()

    # Run RHF
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF calculation did not converge.")

    # Extract integrals in MO basis
    n_orb = mf.mo_coeff.shape[1]
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h1e = np.asarray(h1e, dtype=np.float64)

    # Two-electron integrals: full 4-index tensor in chemist's notation
    eri_mo = ao2mo.full(mol, mf.mo_coeff)
    h2e = ao2mo.restore(1, eri_mo, n_orb).astype(np.float64)

    n_electrons = mol.nelectron
    n_alpha = (n_electrons + spin) // 2
    n_beta = (n_electrons - spin) // 2

    return MolecularIntegrals(
        h1e=h1e,
        h2e=h2e,
        nuclear_repulsion=float(mol.energy_nuc()),
        n_electrons=n_electrons,
        n_orbitals=n_orb,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )
