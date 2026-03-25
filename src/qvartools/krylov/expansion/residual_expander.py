"""
residual_expander --- Residual-based iterative basis expansion
==============================================================

Provides :class:`ResidualBasedExpander`, which identifies missing
configurations via the residual vector
:math:`|r\\rangle = (H - E)|\\Phi\\rangle` and adds those with the
largest residual components to the basis.

Classes
-------
ResidualBasedExpander
    Iterative basis expansion driven by residual analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.krylov.expansion.residual_config import (
    ResidualExpansionConfig,
    _diagonalise_in_basis,
    _generate_candidate_configs,
)

__all__ = [
    "ResidualBasedExpander",
]

logger = logging.getLogger(__name__)


class ResidualBasedExpander:
    r"""Iterative basis expansion driven by residual analysis.

    Given a current basis and its ground-state eigenpair :math:`(E, |\Phi\rangle)`,
    computes the residual :math:`r_x = \langle x | (H - E) | \Phi \rangle`
    for candidate configurations :math:`|x\rangle` and adds those with the
    largest :math:`|r_x|` to the basis.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The system Hamiltonian.
    config : ResidualExpansionConfig
        Expansion hyperparameters.

    Attributes
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian instance.
    config : ResidualExpansionConfig
        The configuration dataclass.

    Examples
    --------
    >>> expander = ResidualBasedExpander(hamiltonian, ResidualExpansionConfig())
    >>> expanded, stats = expander.expand_basis(basis, energy, eigvec)
    >>> stats["iterations"]
    3
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        config: ResidualExpansionConfig,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.config = config

    def _diagonalize(
        self, basis: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        """Solve the eigenvalue problem in the current basis.

        Parameters
        ----------
        basis : torch.Tensor
            Basis configurations, shape ``(n_basis, num_sites)``.

        Returns
        -------
        energy : float
            Lowest eigenvalue.
        eigenvector : np.ndarray
            Corresponding eigenvector in the basis representation.
        """
        return _diagonalise_in_basis(self.hamiltonian, basis)

    def _find_residual_configs(
        self,
        basis: torch.Tensor,
        energy: float,
        eigenvector: np.ndarray,
    ) -> torch.Tensor:
        r"""Find important missing configurations via residual analysis.

        Computes :math:`r_x = \langle x | H - E | \Phi \rangle` for each
        candidate configuration :math:`|x\rangle` connected to the basis.

        Parameters
        ----------
        basis : torch.Tensor
            Current basis configurations, shape ``(n_basis, num_sites)``.
        energy : float
            Current ground-state energy estimate.
        eigenvector : np.ndarray
            Current eigenvector in the basis representation,
            shape ``(n_basis,)``.

        Returns
        -------
        torch.Tensor
            New configurations to add, shape ``(n_new, num_sites)``,
            sorted by decreasing residual magnitude.
        """
        candidates = _generate_candidate_configs(self.hamiltonian, basis)
        if candidates.shape[0] == 0:
            return candidates

        # Compute residual components: r_x = sum_i c_i * <x|H|b_i> - E * 0
        # Since candidates are not in the basis, <x|Phi> = 0, so
        # r_x = sum_i c_i * <x|H|b_i>
        h_cross = self.hamiltonian.matrix_elements(candidates, basis)
        h_cross_np = h_cross.detach().numpy().astype(np.float64)

        # r_x = h_cross @ c  (the (H-E)|Phi> projected onto candidates)
        residuals = h_cross_np @ eigenvector

        abs_residuals = np.abs(residuals)

        # Filter by threshold
        mask = abs_residuals >= self.config.residual_threshold
        if not mask.any():
            return torch.zeros(0, basis.shape[1], dtype=torch.int64)

        valid_indices = np.where(mask)[0]
        valid_residuals = abs_residuals[valid_indices]

        # Sort by decreasing residual magnitude
        sorted_order = np.argsort(-valid_residuals)
        selected = valid_indices[sorted_order[: self.config.max_configs_per_iter]]

        return candidates[selected]

    def expand_basis(
        self,
        current_basis: torch.Tensor,
        energy: float,
        eigenvector: np.ndarray,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Iteratively expand the basis using residual analysis.

        Parameters
        ----------
        current_basis : torch.Tensor
            Starting basis configurations, shape ``(n_basis, num_sites)``.
        energy : float
            Current ground-state energy estimate.
        eigenvector : np.ndarray
            Current eigenvector in the basis representation.

        Returns
        -------
        expanded_basis : torch.Tensor
            Expanded basis configurations.
        stats : dict
            Expansion statistics with keys:

            - ``"iterations"`` : int — number of expansion iterations.
            - ``"initial_energy"`` : float — energy before expansion.
            - ``"final_energy"`` : float — energy after expansion.
            - ``"energy_improvement_mha"`` : float — total improvement in mHa.
            - ``"basis_sizes"`` : list of int — basis size at each iteration.
            - ``"energies"`` : list of float — energy at each iteration.
            - ``"configs_added"`` : list of int — configs added per iteration.
        """
        basis = current_basis.clone()
        current_energy = energy
        current_eigvec = eigenvector.copy()

        initial_energy = current_energy
        stagnation_count = 0
        basis_sizes: List[int] = [basis.shape[0]]
        energies: List[float] = [current_energy]
        configs_added: List[int] = []

        for iteration in range(self.config.max_iterations):
            if basis.shape[0] >= self.config.max_basis_size:
                logger.info(
                    "Reached max_basis_size=%d, stopping expansion.",
                    self.config.max_basis_size,
                )
                break

            new_configs = self._find_residual_configs(
                basis, current_energy, current_eigvec
            )

            if new_configs.shape[0] == 0:
                logger.info("No new residual configs found, stopping.")
                break

            # Respect max_basis_size
            space_left = self.config.max_basis_size - basis.shape[0]
            if new_configs.shape[0] > space_left:
                new_configs = new_configs[:space_left]

            basis = torch.cat([basis, new_configs], dim=0)
            configs_added.append(new_configs.shape[0])

            # Re-diagonalise
            new_energy, new_eigvec = self._diagonalize(basis)

            improvement_mha = (current_energy - new_energy) * 1000.0
            logger.info(
                "Iteration %d: basis_size=%d, E=%.10f, dE=%.4f mHa",
                iteration,
                basis.shape[0],
                new_energy,
                improvement_mha,
            )

            basis_sizes.append(basis.shape[0])
            energies.append(new_energy)

            # Check for stagnation
            if improvement_mha < self.config.min_energy_improvement_mha:
                stagnation_count += 1
                if stagnation_count >= self.config.stagnation_patience:
                    logger.info(
                        "Energy stagnation for %d iterations, stopping.",
                        stagnation_count,
                    )
                    break
            else:
                stagnation_count = 0

            current_energy = new_energy
            current_eigvec = new_eigvec

        total_improvement_mha = (initial_energy - current_energy) * 1000.0

        stats: Dict[str, Any] = {
            "iterations": len(configs_added),
            "initial_energy": initial_energy,
            "final_energy": current_energy,
            "energy_improvement_mha": total_improvement_mha,
            "basis_sizes": basis_sizes,
            "energies": energies,
            "configs_added": configs_added,
        }

        return basis, stats
