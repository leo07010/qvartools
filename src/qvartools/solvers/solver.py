"""
base --- Abstract solver interface and result dataclass
=======================================================

Defines the :class:`Solver` ABC that every concrete solver must implement,
together with the :class:`SolverResult` immutable dataclass that standardises
solver outputs across all methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from qvartools.hamiltonians.hamiltonian import Hamiltonian

__all__ = [
    "SolverResult",
    "Solver",
]


# ---------------------------------------------------------------------------
# SolverResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverResult:
    """Immutable container for solver output.

    Parameters
    ----------
    energy : float or None
        Ground-state energy estimate in Hartree, or ``None`` when the
        solver fails or skips the computation.
    diag_dim : int
        Dimension of the diagonalisation subspace (number of basis
        configurations used).
    wall_time : float
        Wall-clock time in seconds for the full solve.
    method : str
        Human-readable solver method name (e.g. ``"FCI"``, ``"SQD"``).
    converged : bool
        Whether the solver converged to the requested tolerance.
    metadata : dict
        Additional method-specific information (e.g. iteration history,
        basis size per step, training metrics).

    Attributes
    ----------
    energy : float or None
    diag_dim : int
    wall_time : float
    method : str
    converged : bool
    metadata : dict

    Examples
    --------
    >>> result = SolverResult(
    ...     diag_dim=100, wall_time=1.5, method="SQD", converged=True,
    ...     energy=-1.137,
    ... )
    >>> result.energy
    -1.137
    """

    diag_dim: int
    wall_time: float
    method: str
    converged: bool
    energy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        energy_str = f"{self.energy:.10f}" if self.energy is not None else "None"
        return (
            f"SolverResult(method={self.method!r}, energy={energy_str}, "
            f"diag_dim={self.diag_dim}, wall_time={self.wall_time:.2f}s, "
            f"converged={self.converged})"
        )


# ---------------------------------------------------------------------------
# Solver ABC
# ---------------------------------------------------------------------------


class Solver(ABC):
    """Abstract base class for all quantum chemistry solvers.

    Every subclass must implement :meth:`solve`, which takes a Hamiltonian
    and a molecular information dictionary and returns a :class:`SolverResult`.

    The ``mol_info`` dictionary is expected to contain at least:

    - ``"name"`` : str -- molecule name.
    - ``"n_qubits"`` : int -- number of qubits (spin-orbitals).
    - ``"basis"`` : str -- Gaussian basis set.
    - ``"geometry"`` : list -- atomic geometry.
    - ``"charge"`` : int -- net molecular charge.
    - ``"spin"`` : int -- spin multiplicity minus one (2S).
    """

    @abstractmethod
    def solve(
        self, hamiltonian: Hamiltonian, mol_info: Dict[str, Any]
    ) -> SolverResult:
        """Compute the ground-state energy.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            The molecular Hamiltonian to diagonalise.
        mol_info : dict
            Molecular metadata dictionary with keys ``name``, ``n_qubits``,
            ``basis``, ``geometry``, ``charge``, ``spin``.

        Returns
        -------
        SolverResult
            Result containing energy, timing, and convergence information.
        """
