"""
integrals — Molecular integral container and PySCF computation
==============================================================

Provides the ``MolecularIntegrals`` frozen dataclass that holds one- and
two-electron integrals together with molecule metadata, and the
``compute_molecular_integrals`` helper that runs RHF via PySCF and
returns a populated ``MolecularIntegrals`` instance.

Integral caching
----------------
``compute_molecular_integrals`` supports disk caching via
``use_cache=True`` (default).  Integrals are stored as compressed NumPy
``.npz`` files under ``~/.cache/qvartools/``, keyed by a SHA-256 hash of
the geometry, basis, charge and spin.  This avoids redundant PySCF SCF
calculations across sessions.

FCIDUMP support
---------------
``load_fcidump_integrals`` reads a FCIDUMP file (Knowles & Handy 1989)
and returns a ``MolecularIntegrals`` instance.  This is the recommended
way to load large pre-computed active-space integrals such as the
[2Fe-2S] and [4Fe-4S] benchmarks from Li & Chan (JCTC 2017).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "MATRIX_ELEMENT_TOL",
    "MolecularIntegrals",
    "compute_molecular_integrals",
    "load_fcidump_integrals",
]

MATRIX_ELEMENT_TOL: float = 1e-12
"""float : Absolute tolerance below which matrix elements are treated as zero."""

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "qvartools"


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
# Internal cache helpers
# ---------------------------------------------------------------------------


def _integral_cache_key(
    geometry: list,
    basis: str,
    charge: int,
    spin: int,
) -> str:
    """Return a 16-character SHA-256 cache key for the given molecular spec."""
    geo_str = json.dumps(
        [(sym, tuple(round(x, 8) for x in coords)) for sym, coords in geometry],
        sort_keys=True,
    )
    raw = f"{geo_str}|{basis.lower()}|{charge}|{spin}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cached_integrals(key: str) -> Optional[MolecularIntegrals]:
    """Load MolecularIntegrals from cache. Returns None on miss or error."""
    cache_file = _CACHE_DIR / f"{key}.npz"
    if not cache_file.exists():
        return None
    try:
        data = np.load(str(cache_file), allow_pickle=False)
        return MolecularIntegrals(
            h1e=data["h1e"],
            h2e=data["h2e"],
            nuclear_repulsion=float(data["nuclear_repulsion"]),
            n_electrons=int(data["n_electrons"]),
            n_orbitals=int(data["n_orbitals"]),
            n_alpha=int(data["n_alpha"]),
            n_beta=int(data["n_beta"]),
        )
    except Exception as exc:
        logger.warning("Corrupted cache file %s, deleting: %s", cache_file, exc)
        cache_file.unlink(missing_ok=True)
        return None


def _save_cached_integrals(key: str, integrals: MolecularIntegrals) -> None:
    """Save MolecularIntegrals to cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{key}.npz"
    np.savez_compressed(
        str(cache_file),
        h1e=integrals.h1e.astype(np.float64),
        h2e=integrals.h2e.astype(np.float64),
        nuclear_repulsion=np.float64(integrals.nuclear_repulsion),
        n_electrons=np.int64(integrals.n_electrons),
        n_orbitals=np.int64(integrals.n_orbitals),
        n_alpha=np.int64(integrals.n_alpha),
        n_beta=np.int64(integrals.n_beta),
    )


# ---------------------------------------------------------------------------
# compute_molecular_integrals (PySCF)
# ---------------------------------------------------------------------------


def compute_molecular_integrals(
    geometry: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    use_cache: bool = True,
) -> MolecularIntegrals:
    """Run RHF with PySCF and extract molecular integrals.

    Results are cached to ``~/.cache/qvartools/`` as compressed ``.npz``
    files keyed by a SHA-256 hash of *(geometry, basis, charge, spin)*.
    Subsequent calls with the same arguments load from cache instead of
    re-running PySCF, saving significant wall-clock time.

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
    use_cache : bool, optional
        If ``True`` (default), load from disk cache when available and
        save newly computed integrals to cache.

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
    # --- Cache lookup ---
    if use_cache:
        key = _integral_cache_key(geometry, basis, charge, spin)
        cached = _load_cached_integrals(key)
        if cached is not None:
            logger.info("[IntegralCache] Loaded integrals from cache (%s)", key)
            return cached

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

    result = MolecularIntegrals(
        h1e=h1e,
        h2e=h2e,
        nuclear_repulsion=float(mol.energy_nuc()),
        n_electrons=n_electrons,
        n_orbitals=n_orb,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    # --- Cache save ---
    if use_cache:
        try:
            _save_cached_integrals(key, result)
            logger.info("[IntegralCache] Saved integrals to cache (%s)", key)
        except Exception as exc:
            logger.warning("[IntegralCache] Failed to save integrals: %s", exc)

    return result


# ---------------------------------------------------------------------------
# load_fcidump_integrals
# ---------------------------------------------------------------------------


def load_fcidump_integrals(fcidump_path: str | Path) -> MolecularIntegrals:
    """Load molecular integrals from a FCIDUMP file.

    FCIDUMP (Knowles & Handy 1989) is a standard plain-text format for
    storing one- and two-electron integrals.  It is produced by MOLPRO,
    PySCF, ORCA and many other quantum-chemistry codes.

    This function is the recommended way to load large pre-computed
    active-space integrals — for example, the [2Fe-2S] CAS(30e,20o) and
    [4Fe-4S] CAS(54e,36o) integrals from Li & Chan (JCTC 2017), which are
    used by IBM's SQD experiments (Robledo-Moreno et al. 2024).

    Parameters
    ----------
    fcidump_path : str or Path
        Path to the FCIDUMP file.

    Returns
    -------
    MolecularIntegrals
        Integrals compatible with :class:`MolecularHamiltonian`.

    Raises
    ------
    ImportError
        If PySCF is not installed.
    FileNotFoundError
        If the FCIDUMP file does not exist.

    Notes
    -----
    FCIDUMP-loaded integrals are **not** saved to the integral cache
    because the FCIDUMP file itself already serves as the persisted
    representation.

    Examples
    --------
    >>> from qvartools.hamiltonians.integrals import load_fcidump_integrals
    >>> mi = load_fcidump_integrals("path/to/2Fe2S.fcidump")  # doctest: +SKIP
    >>> mi.n_orbitals  # doctest: +SKIP
    20
    """
    fcidump_path = Path(fcidump_path)
    if not fcidump_path.exists():
        raise FileNotFoundError(f"FCIDUMP file not found: {fcidump_path}")

    try:
        from pyscf import ao2mo as _ao2mo
        from pyscf.tools import fcidump as pyscf_fcidump
    except ImportError as exc:
        raise ImportError(
            "PySCF is required to read FCIDUMP files. "
            "Install it with: pip install pyscf"
        ) from exc

    data = pyscf_fcidump.read(str(fcidump_path))

    norb = int(data["NORB"])
    nelec = data["NELEC"]
    ms2 = int(data.get("MS2", 0))

    # NELEC may be an int (total) or a sequence (n_alpha, n_beta)
    if isinstance(nelec, (list, tuple)):
        n_alpha, n_beta = int(nelec[0]), int(nelec[1])
        n_electrons = n_alpha + n_beta
    else:
        n_electrons = int(nelec)
        n_alpha = (n_electrons + ms2) // 2
        n_beta = (n_electrons - ms2) // 2

    h1e = np.asarray(data["H1"], dtype=np.float64).reshape(norb, norb)
    # H2 is stored with 4-fold or 8-fold symmetry; restore to full 4-index tensor
    h2e = _ao2mo.restore(1, np.asarray(data["H2"], dtype=np.float64), norb)
    ecore = float(data.get("ECORE", 0.0))

    return MolecularIntegrals(
        h1e=h1e,
        h2e=h2e,
        nuclear_repulsion=ecore,
        n_electrons=n_electrons,
        n_orbitals=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )
