"""
selected_ci_expander --- CIPSI-style selected-CI basis expansion
================================================================

Provides :class:`SelectedCIExpander`, which uses second-order perturbative
importance (CIPSI-style) to select the most important configurations for
basis enrichment.

Classes
-------
SelectedCIExpander
    CIPSI-style selected-CI basis expansion using perturbative importance.
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
    "SelectedCIExpander",
]

logger = logging.getLogger(__name__)


class SelectedCIExpander:
    r"""CIPSI-style selected-CI basis expansion.

    Uses second-order perturbative importance to select the most
    significant configurations for basis enrichment:

    .. math::

        \varepsilon_x = \frac{|\langle x | H | \Phi \rangle|^2}{|E - E_x|}

    where :math:`E_x = \langle x | H | x \rangle` is the diagonal
    Hamiltonian element and :math:`E` is the current variational energy.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The system Hamiltonian.
    config : ResidualExpansionConfig
        Expansion hyperparameters (shared with residual expansion).

    Attributes
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian instance.
    config : ResidualExpansionConfig
        The configuration dataclass.

    Examples
    --------
    >>> expander = SelectedCIExpander(hamiltonian, ResidualExpansionConfig())
    >>> expanded, stats = expander.expand_basis(basis, energy, eigvec)
    >>> "pt2_corrections" in stats
    True
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

    def _compute_perturbative_importance(
        self,
        basis: torch.Tensor,
        energy: float,
        eigenvector: np.ndarray,
        candidates: torch.Tensor,
    ) -> np.ndarray:
        r"""Compute CIPSI-style perturbative importance for candidates.

        Parameters
        ----------
        basis : torch.Tensor
            Current basis configurations, shape ``(n_basis, num_sites)``.
        energy : float
            Current variational energy.
        eigenvector : np.ndarray
            Current eigenvector in the basis representation.
        candidates : torch.Tensor
            Candidate configurations, shape ``(n_candidates, num_sites)``.

        Returns
        -------
        np.ndarray
            Perturbative importance :math:`\varepsilon_x` for each candidate,
            shape ``(n_candidates,)``.
        """
        # Compute <x|H|Phi> = sum_i c_i <x|H|b_i>
        h_cross = self.hamiltonian.matrix_elements(candidates, basis)
        h_cross_np = h_cross.detach().numpy().astype(np.float64)
        coupling = h_cross_np @ eigenvector  # shape (n_candidates,)

        # Compute diagonal elements E_x = <x|H|x>
        diag_elements = np.array(
            [
                float(self.hamiltonian.diagonal_element(candidates[i]))
                for i in range(candidates.shape[0])
            ],
            dtype=np.float64,
        )

        # epsilon_x = |<x|H|Phi>|^2 / |E - E_x|
        denominator = np.abs(energy - diag_elements)
        # Avoid division by zero with a small epsilon
        denominator = np.maximum(denominator, 1e-12)

        importance = np.abs(coupling) ** 2 / denominator
        return importance

    def expand_basis(
        self,
        current_basis: torch.Tensor,
        energy: float,
        eigenvector: np.ndarray,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Iteratively expand the basis using CIPSI-style selection.

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
            - ``"pt2_corrections"`` : list of float — Epstein--Nesbet PT2
              correction at each iteration.
        """
        basis = current_basis.clone()
        current_energy = energy
        current_eigvec = eigenvector.copy()

        initial_energy = current_energy
        stagnation_count = 0
        basis_sizes: List[int] = [basis.shape[0]]
        energies: List[float] = [current_energy]
        configs_added: List[int] = []
        pt2_corrections: List[float] = []

        for iteration in range(self.config.max_iterations):
            if basis.shape[0] >= self.config.max_basis_size:
                logger.info(
                    "Reached max_basis_size=%d, stopping expansion.",
                    self.config.max_basis_size,
                )
                break

            candidates = _generate_candidate_configs(self.hamiltonian, basis)
            if candidates.shape[0] == 0:
                logger.info("No candidate configs found, stopping.")
                break

            importance = self._compute_perturbative_importance(
                basis, current_energy, current_eigvec, candidates
            )

            # PT2 correction is the sum of all importances
            pt2_correction = float(np.sum(importance))
            pt2_corrections.append(pt2_correction)

            # Filter by threshold (re-interpret residual_threshold for importance)
            mask = importance >= self.config.residual_threshold
            if not mask.any():
                logger.info("No candidates above importance threshold, stopping.")
                break

            valid_indices = np.where(mask)[0]
            valid_importance = importance[valid_indices]

            # Sort by decreasing importance
            sorted_order = np.argsort(-valid_importance)
            n_to_add = min(
                self.config.max_configs_per_iter,
                len(sorted_order),
                self.config.max_basis_size - basis.shape[0],
            )
            selected = valid_indices[sorted_order[:n_to_add]]
            new_configs = candidates[selected]

            basis = torch.cat([basis, new_configs], dim=0)
            configs_added.append(new_configs.shape[0])

            # Re-diagonalise
            new_energy, new_eigvec = self._diagonalize(basis)

            improvement_mha = (current_energy - new_energy) * 1000.0
            logger.info(
                "Iteration %d: basis_size=%d, E=%.10f, dE=%.4f mHa, PT2=%.6f",
                iteration,
                basis.shape[0],
                new_energy,
                improvement_mha,
                pt2_correction,
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
            "pt2_corrections": pt2_corrections,
        }

        return basis, stats
