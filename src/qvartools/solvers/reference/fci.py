"""
fci --- Full Configuration Interaction solver
==============================================

Implements :class:`FCISolver`, which computes the exact ground-state
energy via full configuration interaction.  Uses PySCF's FCI module
when available; falls back to dense diagonalisation of the Hamiltonian
matrix for small systems.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.solvers.solver import Solver, SolverResult

__all__ = [
    "FCISolver",
]

logger = logging.getLogger(__name__)


class FCISolver(Solver):
    """Full configuration interaction solver.

    Attempts to use PySCF's FCI module for molecular Hamiltonians.
    When PySCF is unavailable or the Hamiltonian is not molecular,
    falls back to dense exact diagonalisation via
    :meth:`~qvartools.hamiltonians.hamiltonian.Hamiltonian.exact_ground_state`.

    Parameters
    ----------
    max_configs : int, optional
        Maximum number of configurations (Hilbert-space dimension) for
        which dense diagonalisation is attempted (default ``500_000``).
        Systems exceeding this limit raise a ``RuntimeError`` when PySCF
        is unavailable.

    Attributes
    ----------
    max_configs : int
        Configuration limit for dense fallback.

    Examples
    --------
    >>> solver = FCISolver()
    >>> result = solver.solve(hamiltonian, mol_info)
    >>> result.energy
    -1.1373060356
    """

    def __init__(self, max_configs: int = 500_000) -> None:
        if max_configs < 1:
            raise ValueError(f"max_configs must be >= 1, got {max_configs}")
        self.max_configs: int = max_configs

    def solve(
        self, hamiltonian: Hamiltonian, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Compute the FCI ground-state energy.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            The molecular Hamiltonian.
        mol_info : dict
            Molecular metadata (must contain ``"name"``).

        Returns
        -------
        SolverResult
            FCI energy result.

        Raises
        ------
        RuntimeError
            If the Hilbert-space dimension exceeds ``max_configs`` and
            PySCF is not available.
        """
        t_start = time.perf_counter()

        energy, diag_dim, converged, metadata = self._try_pyscf_fci(
            hamiltonian, mol_info
        )

        if energy is None:
            energy, diag_dim, converged, metadata = self._dense_fallback(
                hamiltonian
            )

        wall_time = time.perf_counter() - t_start

        logger.info(
            "FCISolver [%s]: energy=%.10f, dim=%d, time=%.2fs",
            mol_info.get("name", "unknown"),
            energy,
            diag_dim,
            wall_time,
        )

        return SolverResult(
            energy=energy,
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="FCI",
            converged=converged,
            metadata=metadata,
        )

    def _try_pyscf_fci(
        self, hamiltonian: Hamiltonian, mol_info: Dict[str, Any]
    ) -> tuple:
        """Attempt FCI via PySCF.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            Must have an ``integrals`` attribute for PySCF FCI.
        mol_info : dict
            Molecular metadata.

        Returns
        -------
        tuple
            ``(energy, diag_dim, converged, metadata)`` or
            ``(None, 0, False, {})`` if PySCF is unavailable or the
            Hamiltonian is not molecular.
        """
        if not hasattr(hamiltonian, "integrals"):
            return None, 0, False, {}

        try:
            from pyscf import fci, gto, scf
        except ImportError:
            logger.info("PySCF not available; falling back to dense FCI.")
            return None, 0, False, {}

        integrals = hamiltonian.integrals
        geometry = mol_info.get("geometry", [])
        basis = mol_info.get("basis", "sto-3g")
        charge = mol_info.get("charge", 0)
        spin = mol_info.get("spin", 0)

        mol = gto.Mole()
        mol.atom = [(atom, coord) for atom, coord in geometry]
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.unit = "Angstrom"
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        cisolver = fci.FCI(mf)
        e_fci, ci_vec = cisolver.kernel()

        n_orb = integrals.n_orbitals
        n_alpha = integrals.n_alpha
        n_beta = integrals.n_beta

        from math import comb

        diag_dim = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

        metadata: Dict[str, Any] = {
            "pyscf_converged": mf.converged,
            "n_orbitals": n_orb,
            "n_alpha": n_alpha,
            "n_beta": n_beta,
        }

        return float(e_fci), diag_dim, True, metadata

    def _dense_fallback(
        self, hamiltonian: Hamiltonian
    ) -> tuple:
        """Fall back to dense exact diagonalisation.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            The Hamiltonian to diagonalise.

        Returns
        -------
        tuple
            ``(energy, diag_dim, converged, metadata)``.

        Raises
        ------
        RuntimeError
            If the Hilbert-space dimension exceeds ``max_configs``.
        """
        diag_dim = hamiltonian.hilbert_dim
        if diag_dim > self.max_configs:
            raise RuntimeError(
                f"Dense FCI requires {diag_dim} configurations, which exceeds "
                f"max_configs={self.max_configs}. Install PySCF for large systems."
            )

        logger.info("Using dense diagonalisation (dim=%d).", diag_dim)
        energy, _ = hamiltonian.exact_ground_state()

        metadata: Dict[str, Any] = {"fallback": "dense_diag"}
        return energy, diag_dim, True, metadata
