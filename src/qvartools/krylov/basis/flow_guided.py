"""
flow_guided --- Flow-guided classical Krylov diagonalization
=============================================================

Provides :class:`FlowGuidedKrylovDiag`, a classical Krylov variant that
seeds the basis with configurations from a normalizing-flow sampler for
accelerated convergence.

Classes
-------
FlowGuidedKrylovDiag
    Classical Krylov diagonalization with normalizing-flow basis seeding.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.krylov.basis.skqd import (
    ClassicalKrylovDiagonalization,
    SKQDConfig,
    _build_projected_matrices,
    _solve_generalised_eigenproblem,
)

__all__ = [
    "FlowGuidedKrylovDiag",
    # Deprecated alias (remove in v0.1.0)
    "FlowGuidedSKQD",
]

logger = logging.getLogger(__name__)


class FlowGuidedKrylovDiag:
    r"""SKQD with normalizing-flow basis seeding.

    Combines configurations obtained from a normalizing-flow sampler with
    the Krylov subspace to accelerate convergence.  The NF basis provides
    an informed starting set of configurations, and Krylov expansion
    systematically enriches it.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The system Hamiltonian.
    config : SKQDConfig
        Algorithm hyperparameters.
    nf_basis : torch.Tensor
        Configurations from the normalizing-flow sampler,
        shape ``(n_nf, num_sites)``.
    nf_basis_weights : torch.Tensor or None, optional
        Importance weights for the NF configurations.  If ``None``,
        uniform weights are assumed.
    initial_state : np.ndarray or None, optional
        Initial state for the Krylov expansion.  If ``None``, a default
        Hartree--Fock-like state is constructed.

    Examples
    --------
    >>> skqd = FlowGuidedKrylovDiag(hamiltonian, SKQDConfig(), nf_basis=nf_configs)
    >>> results = skqd.run_with_nf(progress=True)
    >>> results["energy"]
    -1.137
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        config: SKQDConfig,
        nf_basis: torch.Tensor,
        nf_basis_weights: torch.Tensor | None = None,
        initial_state: np.ndarray | None = None,
    ) -> None:
        if nf_basis.ndim != 2:
            raise ValueError(
                f"nf_basis must be 2-D (n_configs, num_sites), got shape {nf_basis.shape}"
            )
        if nf_basis.shape[1] != hamiltonian.num_sites:
            raise ValueError(
                f"nf_basis num_sites ({nf_basis.shape[1]}) does not match "
                f"hamiltonian ({hamiltonian.num_sites})"
            )

        self.hamiltonian = hamiltonian
        self.config = config
        self.nf_basis = nf_basis
        self.nf_basis_weights = nf_basis_weights

        # Delegate Krylov machinery to the base solver
        self._skqd = ClassicalKrylovDiagonalization(hamiltonian, config, initial_state)

    def _build_matrices(self, configs: torch.Tensor) -> tuple:
        """Build projected Hamiltonian and overlap matrices.

        Delegates to ``self._skqd.extract_projected_submatrix`` (fast
        O(n^2) index extraction from the precomputed dense Hamiltonian)
        when the SKQD solver has a subspace index.  Falls back to the
        standard :func:`_build_projected_matrices` otherwise.

        Parameters
        ----------
        configs : torch.Tensor
            Basis configurations, shape ``(n_basis, num_sites)``.

        Returns
        -------
        h_proj : np.ndarray
            Projected Hamiltonian, shape ``(n_basis, n_basis)``.
        s_proj : np.ndarray
            Overlap matrix, shape ``(n_basis, n_basis)``.
        """
        if self._skqd._subspace_hash_to_idx:
            return self._skqd.extract_projected_submatrix(configs)
        return _build_projected_matrices(self.hamiltonian, configs)

    def run_with_nf(self, progress: bool = False) -> dict[str, Any]:
        """Run SKQD with NF basis seeding.

        Parameters
        ----------
        progress : bool, optional
            If ``True``, log progress at each Krylov expansion step.

        Returns
        -------
        dict
            Results dictionary with keys:

            - ``"energy"`` : float — best stable ground-state energy estimate.
            - ``"eigenvalues"`` : np.ndarray — all computed eigenvalues from
              the final projection.
            - ``"basis_size"`` : int — final number of basis configurations.
            - ``"krylov_dim"`` : int — number of Krylov vectors used.
            - ``"energies_per_step"`` : list of float — ground-state energy
              estimate after each expansion step.
            - ``"nf_energy"`` : float — energy from the NF-only basis.
            - ``"basis_configs"`` : torch.Tensor — final basis configurations.
        """
        # ---- Step 1: start with NF basis ----
        all_configs = torch.unique(self.nf_basis, dim=0)
        energies_per_step: list[float] = []

        h_proj, s_proj = self._build_matrices(all_configs)
        eigenvalues, _ = _solve_generalised_eigenproblem(
            h_proj,
            s_proj,
            self.config.num_eigenvalues,
            self.config.regularization,
        )
        nf_energy = float(eigenvalues[0])
        energies_per_step.append(nf_energy)

        if progress:
            logger.info(
                "NF-only basis: size=%d, E0=%.10f",
                all_configs.shape[0],
                nf_energy,
            )

        # ---- Step 2: iteratively add Krylov states ----
        best_energy = nf_energy

        for k in range(self.config.max_krylov_dim):
            krylov_state = self._skqd._compute_krylov_state(k)
            sampled = self._skqd._sample_from_state(
                krylov_state, self.config.shots_per_krylov
            )

            # Merge with existing basis
            combined = torch.cat([all_configs, sampled], dim=0)
            all_configs = torch.unique(combined, dim=0)

            # Build and solve projected problem
            h_proj, s_proj = self._build_matrices(all_configs)
            eigenvalues, _ = _solve_generalised_eigenproblem(
                h_proj,
                s_proj,
                self.config.num_eigenvalues,
                self.config.regularization,
            )

            step_energy = float(eigenvalues[0])
            energies_per_step.append(step_energy)

            if step_energy < best_energy:
                best_energy = step_energy

            if progress:
                logger.info(
                    "Krylov k=%d: basis_size=%d, E0=%.10f",
                    k,
                    all_configs.shape[0],
                    step_energy,
                )

        # ---- Step 3: final projection ----
        h_proj, s_proj = self._build_matrices(all_configs)
        eigenvalues, _ = _solve_generalised_eigenproblem(
            h_proj,
            s_proj,
            self.config.num_eigenvalues,
            self.config.regularization,
        )

        final_energy = float(eigenvalues[0])
        if final_energy < best_energy:
            best_energy = final_energy

        return {
            "energy": best_energy,
            "eigenvalues": eigenvalues,
            "basis_size": all_configs.shape[0],
            "krylov_dim": self.config.max_krylov_dim,
            "energies_per_step": energies_per_step,
            "nf_energy": nf_energy,
            "basis_configs": all_configs,
        }


# ---------------------------------------------------------------------------
# Deprecated alias (remove in v0.1.0)
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    """Emit DeprecationWarning for old name."""
    import warnings

    if name == "FlowGuidedSKQD":
        warnings.warn(
            "FlowGuidedSKQD is deprecated, use FlowGuidedKrylovDiag instead. "
            "The old name will be removed in v0.1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FlowGuidedKrylovDiag
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
