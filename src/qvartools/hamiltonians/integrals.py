"""
integrals — Molecular integral container and PySCF computation
==============================================================

Provides the ``MolecularIntegrals`` frozen dataclass that holds one- and
two-electron integrals together with molecule metadata, and the
``compute_molecular_integrals`` helper that runs RHF via PySCF and
returns a populated ``MolecularIntegrals`` instance.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MATRIX_ELEMENT_TOL",
    "MolecularIntegrals",
    "cached_compute_molecular_integrals",
    "clear_integral_cache",
    "compute_molecular_integrals",
    "get_integral_cache",
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
    geometry: list[tuple[str, tuple[float, float, float]]],
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


# ---------------------------------------------------------------------------
# Persistent cache via joblib
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("QVARTOOLS_CACHE_DIR", os.path.expanduser("~/.cache/qvartools")),
    "integrals",
)


def get_integral_cache(
    cache_dir: str | None = None,
) -> Callable[..., MolecularIntegrals]:
    """Return a cached version of :func:`compute_molecular_integrals`.

    Uses ``joblib.Memory`` for transparent disk-based caching of PySCF
    integral computations.  Repeated calls with the same arguments
    return instantly from disk.

    Parameters
    ----------
    cache_dir : str or None, optional
        Directory for cached results.  Defaults to
        ``~/.cache/qvartools/integrals`` (overridable via
        ``QVARTOOLS_CACHE_DIR`` environment variable).

    Returns
    -------
    callable
        A cached version of ``compute_molecular_integrals`` with the
        same signature.
    """
    try:
        from joblib import Memory
    except ImportError as exc:
        raise ImportError(
            "joblib is required for integral caching. "
            "Install it with: pip install joblib"
        ) from exc

    location = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    memory = Memory(location, verbose=0)
    cached_fn = memory.cache(compute_molecular_integrals)
    logger.info("Integral cache enabled at %s", location)
    return cached_fn


# Module-level default cached function (lazy init)
_default_cached_fn: Callable[..., MolecularIntegrals] | None = None


def cached_compute_molecular_integrals(
    geometry: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
) -> MolecularIntegrals:
    """Cached version of :func:`compute_molecular_integrals`.

    Identical interface, but results are persisted to disk via
    ``joblib.Memory``.  The default cache directory is
    ``~/.cache/qvartools/integrals``.

    Parameters
    ----------
    geometry : list of (str, (float, float, float))
        Molecular geometry.
    basis : str, optional
        Basis set name (default ``"sto-3g"``).
    charge : int, optional
        Net charge (default ``0``).
    spin : int, optional
        2S (default ``0``).

    Returns
    -------
    MolecularIntegrals
        Cached or freshly computed integrals.
    """
    global _default_cached_fn  # noqa: PLW0603
    if _default_cached_fn is None:
        _default_cached_fn = get_integral_cache()
    return _default_cached_fn(geometry, basis=basis, charge=charge, spin=spin)


def clear_integral_cache(cache_dir: str | None = None) -> None:
    """Remove all cached integral data.

    Parameters
    ----------
    cache_dir : str or None, optional
        Cache directory to clear.  Defaults to the same directory
        used by :func:`get_integral_cache`.
    """
    location = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    # Safety: refuse to delete directories that don't look like a cache
    if "qvartools" not in location and "cache" not in location.lower():
        raise ValueError(
            f"Refusing to delete '{location}' — path does not contain "
            f"'qvartools' or 'cache'. Pass an explicit cache directory."
        )
    if os.path.isdir(location):
        shutil.rmtree(location)
        logger.info("Integral cache cleared at %s", location)
