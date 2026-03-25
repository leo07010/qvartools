"""
ccsd --- Coupled Cluster Singles and Doubles solver
===================================================

Implements :class:`CCSDSolver`, which computes the ground-state energy
via CCSD using PySCF.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.solvers.solver import Solver, SolverResult

__all__ = [
    "CCSDSolver",
]

logger = logging.getLogger(__name__)


class CCSDSolver(Solver):
    """Coupled cluster singles and doubles solver via PySCF.

    Requires PySCF to be installed.  The solver runs RHF followed by
    CCSD on the molecular geometry specified in ``mol_info``.

    Examples
    --------
    >>> solver = CCSDSolver()
    >>> result = solver.solve(hamiltonian, mol_info)
    >>> result.method
    'CCSD'
    """

    def solve(
        self, hamiltonian: Hamiltonian, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Compute the CCSD ground-state energy.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            The molecular Hamiltonian (used for metadata; PySCF recomputes
            integrals internally).
        mol_info : dict
            Molecular metadata.  Must contain ``"geometry"`` and ``"basis"``.

        Returns
        -------
        SolverResult
            CCSD energy result.

        Raises
        ------
        ImportError
            If PySCF is not installed.
        RuntimeError
            If the RHF or CCSD calculation does not converge.
        """
        try:
            from pyscf import cc, gto, scf
        except ImportError as exc:
            raise ImportError(
                "PySCF is required for CCSDSolver. "
                "Install it with: pip install pyscf"
            ) from exc

        t_start = time.perf_counter()

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
        if not mf.converged:
            raise RuntimeError(
                f"RHF did not converge for {mol_info.get('name', 'unknown')}."
            )

        mycc = cc.CCSD(mf)
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError(
                f"CCSD did not converge for {mol_info.get('name', 'unknown')}."
            )

        e_ccsd = mf.e_tot + mycc.e_corr
        wall_time = time.perf_counter() - t_start

        n_orb = mf.mo_coeff.shape[1]
        n_electrons = mol.nelectron
        diag_dim = n_orb * n_electrons  # approximate active-space dimension

        metadata: Dict[str, Any] = {
            "e_hf": float(mf.e_tot),
            "e_corr": float(mycc.e_corr),
            "n_orbitals": n_orb,
            "n_electrons": n_electrons,
            "rhf_converged": mf.converged,
            "ccsd_converged": mycc.converged,
        }

        logger.info(
            "CCSDSolver [%s]: energy=%.10f, time=%.2fs",
            mol_info.get("name", "unknown"),
            e_ccsd,
            wall_time,
        )

        return SolverResult(
            energy=float(e_ccsd),
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="CCSD",
            converged=mycc.converged,
            metadata=metadata,
        )
