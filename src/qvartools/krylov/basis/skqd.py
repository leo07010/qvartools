"""
skqd --- Sample-Based Krylov Quantum Diagonalization
=====================================================

Implements the Sample-Based Krylov Quantum Diagonalization (SKQD) algorithm,
which constructs a Krylov subspace from time-evolved quantum states, samples
computational-basis configurations from each Krylov vector, and solves a
projected generalized eigenvalue problem in the sampled basis.

Classes
-------
SKQDConfig
    Dataclass holding all hyperparameters for the SKQD algorithm.
SampleBasedKrylovDiagonalization
    Core SKQD solver that builds and diagonalises the projected Hamiltonian.

References
----------
.. [1] Robledo-Moreno et al., "Chemistry beyond exact solutions on a
       quantum-centric supercomputer", arXiv:2405.05068 (2024).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import expm_multiply

from qvartools.hamiltonians.hamiltonian import Hamiltonian

__all__ = [
    "SKQDConfig",
    "SampleBasedKrylovDiagonalization",
    "FlowGuidedSKQD",
]


def __getattr__(name: str):
    """Lazy re-export of FlowGuidedSKQD to avoid circular import."""
    if name == "FlowGuidedSKQD":
        from qvartools.krylov.basis.flow_guided import FlowGuidedSKQD
        return FlowGuidedSKQD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SKQDConfig:
    """Hyperparameters for Sample-Based Krylov Quantum Diagonalization.

    Parameters
    ----------
    max_krylov_dim : int
        Maximum number of Krylov vectors to construct.
    time_step : float
        Time step for real-time evolution ``e^{-iH dt}``.
    total_evolution_time : float
        Total evolution time budget (informational; the algorithm uses
        ``max_krylov_dim * time_step`` Krylov vectors).
    shots_per_krylov : int
        Number of measurement shots per Krylov state for sampling
        computational-basis configurations.
    use_cumulative_basis : bool
        If ``True``, retain samples from all previous Krylov states
        (cumulative union).  If ``False``, only keep the latest set.
    num_eigenvalues : int
        Number of lowest eigenvalues to compute from the projected problem.
    which_eigenvalues : str
        Eigenvalue selection criterion passed to the eigensolver.
        ``"SA"`` = smallest algebraic (default).
    regularization : float
        Tikhonov regularization added to the overlap matrix diagonal to
        stabilise the generalized eigenvalue problem.
    use_gpu : bool
        If ``True``, attempt to use GPU-accelerated linear algebra.

    Examples
    --------
    >>> cfg = SKQDConfig(max_krylov_dim=5, time_step=0.05)
    >>> cfg.max_krylov_dim
    5
    """

    max_krylov_dim: int = 10
    time_step: float = 0.1
    total_evolution_time: float = 1.0
    shots_per_krylov: int = 1000
    use_cumulative_basis: bool = True
    num_eigenvalues: int = 1
    which_eigenvalues: str = "SA"
    regularization: float = 1e-8
    use_gpu: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_krylov_dim < 1:
            raise ValueError(
                f"max_krylov_dim must be >= 1, got {self.max_krylov_dim}"
            )
        if self.time_step <= 0.0:
            raise ValueError(f"time_step must be > 0, got {self.time_step}")
        if self.shots_per_krylov < 1:
            raise ValueError(
                f"shots_per_krylov must be >= 1, got {self.shots_per_krylov}"
            )
        if self.num_eigenvalues < 1:
            raise ValueError(
                f"num_eigenvalues must be >= 1, got {self.num_eigenvalues}"
            )
        if self.regularization < 0.0:
            raise ValueError(
                f"regularization must be >= 0, got {self.regularization}"
            )


# ---------------------------------------------------------------------------
# Helper: build projected matrices
# ---------------------------------------------------------------------------


def _build_projected_matrices(
    hamiltonian: Hamiltonian,
    basis_configs: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build projected H and S matrices in a sampled configuration basis.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The system Hamiltonian.
    basis_configs : torch.Tensor
        Basis configurations, shape ``(n_basis, num_sites)``.
    device : torch.device or None, optional
        Device to use for computation.  If ``None``, uses CUDA when available,
        otherwise CPU.

    Returns
    -------
    h_proj : np.ndarray
        Projected Hamiltonian matrix, shape ``(n_basis, n_basis)``.
    s_proj : np.ndarray
        Overlap matrix, shape ``(n_basis, n_basis)``.  For an orthonormal
        computational-basis subset this is the identity, but we keep the
        general form for consistency with weighted / non-orthogonal bases.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_basis = basis_configs.shape[0]
    s_proj = np.eye(n_basis, dtype=np.float64)

    configs_on_device = basis_configs.to(device)

    # Prefer the symmetric fast path when available
    if hasattr(hamiltonian, "matrix_elements_fast"):
        h_matrix = hamiltonian.matrix_elements_fast(configs_on_device)
    else:
        h_matrix = hamiltonian.matrix_elements(configs_on_device, configs_on_device)

    h_proj = h_matrix.detach().cpu().numpy().astype(np.float64)

    return h_proj, s_proj


def _solve_generalised_eigenproblem(
    h_proj: np.ndarray,
    s_proj: np.ndarray,
    num_eigenvalues: int,
    regularization: float,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigenvalue problem ``H v = E S v``.

    When a CUDA device is available and ``use_gpu`` is ``True``, the problem is
    reduced to a standard eigenvalue problem via Cholesky decomposition and
    solved with :func:`torch.linalg.eigh` on the GPU.  Otherwise, falls back
    to :func:`scipy.linalg.eigh` on the CPU.

    Parameters
    ----------
    h_proj : np.ndarray
        Projected Hamiltonian, shape ``(n, n)``.
    s_proj : np.ndarray
        Overlap matrix, shape ``(n, n)``.
    num_eigenvalues : int
        Number of lowest eigenvalues to return.
    regularization : float
        Tikhonov regularization added to the overlap diagonal.
    use_gpu : bool, optional
        If ``True`` (default), use GPU-accelerated ``torch.linalg.eigh``
        when CUDA is available.  If ``False``, always use scipy on CPU.

    Returns
    -------
    eigenvalues : np.ndarray
        Lowest ``num_eigenvalues`` eigenvalues, shape ``(num_eigenvalues,)``.
    eigenvectors : np.ndarray
        Corresponding eigenvectors, shape ``(n, num_eigenvalues)``.
    """
    n = h_proj.shape[0]
    s_reg = s_proj + regularization * np.eye(n, dtype=np.float64)

    # Symmetrise to guard against floating-point asymmetry
    h_sym = 0.5 * (h_proj + h_proj.T)
    s_sym = 0.5 * (s_reg + s_reg.T)

    if use_gpu and torch.cuda.is_available():
        # GPU path: reduce generalised eigenproblem to standard form via
        # Cholesky decomposition  L L^T = S,  then solve
        # L^{-1} H L^{-T} y = E y,  with x = L^{-T} y.
        device = torch.device("cuda")
        h_t = torch.tensor(h_sym, dtype=torch.float64, device=device)
        s_t = torch.tensor(s_sym, dtype=torch.float64, device=device)

        l_chol = torch.linalg.cholesky(s_t)
        l_inv = torch.linalg.inv(l_chol)
        h_standard = l_inv @ h_t @ l_inv.T

        # Symmetrise to remove numerical noise from the transformation
        h_standard = 0.5 * (h_standard + h_standard.T)

        evals, evecs_y = torch.linalg.eigh(h_standard)

        # Transform eigenvectors back: x = L^{-T} y
        evecs_x = l_inv.T @ evecs_y

        eigenvalues = evals.cpu().numpy()
        eigenvectors = evecs_x.cpu().numpy()
    else:
        # CPU path: scipy generalised eigensolver
        eigenvalues, eigenvectors = scipy_eigh(h_sym, s_sym)

    k = min(num_eigenvalues, len(eigenvalues))
    return eigenvalues[:k], eigenvectors[:, :k]


# ---------------------------------------------------------------------------
# SampleBasedKrylovDiagonalization
# ---------------------------------------------------------------------------


class SampleBasedKrylovDiagonalization:
    r"""Sample-Based Krylov Quantum Diagonalization (SKQD).

    Constructs Krylov states
    :math:`|\psi_k\rangle = (e^{-iH\Delta t})^k |\psi_0\rangle`,
    samples computational-basis configurations from each, and solves a
    projected generalized eigenvalue problem in the sampled basis.

    For molecular Hamiltonians the algorithm restricts to the
    particle-number-conserving subspace.  For spin Hamiltonians the full
    Hilbert space is used.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The system Hamiltonian.
    config : SKQDConfig
        Algorithm hyperparameters.
    initial_state : np.ndarray or None, optional
        Initial state vector in the full (or subspace) Hilbert space.
        If ``None``, a default Hartree--Fock-like state is constructed.

    Attributes
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian instance.
    config : SKQDConfig
        The configuration dataclass.
    h_dense : np.ndarray
        Dense Hamiltonian matrix (in the relevant subspace).
    subspace_configs : torch.Tensor or None
        Particle-conserving configurations, if applicable.

    Examples
    --------
    >>> skqd = SampleBasedKrylovDiagonalization(hamiltonian, SKQDConfig())
    >>> eigenvalues, info = skqd.run()
    >>> info["krylov_dim"]
    10
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        config: SKQDConfig,
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.config = config

        # Detect molecular Hamiltonian
        self._is_molecular = hasattr(hamiltonian, "integrals")

        if self._is_molecular:
            self.subspace_configs = self._setup_particle_conserving_subspace()
            self.subspace_dim = self.subspace_configs.shape[0]
            logger.info(
                "Particle-conserving subspace: %d configurations", self.subspace_dim
            )
            # Build dense H in the subspace
            h_torch = hamiltonian.matrix_elements(
                self.subspace_configs, self.subspace_configs
            )
            self.h_dense = h_torch.detach().cpu().numpy().astype(np.float64)
        else:
            self.subspace_configs = None
            self.subspace_dim = hamiltonian.hilbert_dim
            h_torch = hamiltonian.to_dense()
            self.h_dense = h_torch.detach().cpu().numpy().astype(np.float64)

        # Build hash→index mapping for fast submatrix extraction
        self._subspace_hash_to_idx = self._build_subspace_index()

        # Precompute GPU matrix exponential for Krylov time evolution
        self._exp_dt_gpu: Optional[torch.Tensor] = None
        self._initial_state_gpu: Optional[torch.Tensor] = None

        # Initial state
        if initial_state is not None:
            if initial_state.shape[0] != self.subspace_dim:
                raise ValueError(
                    f"initial_state dimension ({initial_state.shape[0]}) does "
                    f"not match subspace dimension ({self.subspace_dim})"
                )
            self._initial_state = initial_state.astype(np.complex128)
        else:
            self._initial_state = self._default_initial_state()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_initial_state(self) -> np.ndarray:
        """Construct a default initial state (first basis vector).

        Returns
        -------
        np.ndarray
            State vector of shape ``(subspace_dim,)``.
        """
        state = np.zeros(self.subspace_dim, dtype=np.complex128)
        state[0] = 1.0
        return state

    def _build_subspace_index(self) -> Dict[int, int]:
        """Build a hash→index mapping for the subspace configurations.

        Uses the Hamiltonian's ``_config_hash_batch`` when available for
        vectorised hashing, otherwise falls back to a powers-of-2 hash.

        Returns
        -------
        dict
            Mapping from integer config hash to subspace index.
        """
        if self.subspace_configs is None:
            return {}

        configs = self.subspace_configs
        n = configs.shape[0]
        num_sites = configs.shape[1]

        # Use Hamiltonian's batch hash if available
        if hasattr(self.hamiltonian, "_config_hash_batch"):
            hashes = self.hamiltonian._config_hash_batch(configs)
        else:
            powers = torch.tensor(
                [1 << k for k in range(num_sites - 1, -1, -1)],
                dtype=torch.int64,
                device=configs.device,
            )
            hashes = (configs.to(torch.int64) * powers.unsqueeze(0)).sum(dim=-1)

        return {int(hashes[i].item()): i for i in range(n)}

    def _precompute_gpu_time_evolution(self) -> None:
        """Precompute exp(-iH*dt) on GPU for fast Krylov state generation.

        Called lazily on the first ``_compute_krylov_state`` call when
        CUDA is available.  Stores the precomputed matrix exponential
        and the initial state on GPU.
        """
        if not torch.cuda.is_available():
            return

        device = torch.device("cuda")
        h_gpu = torch.tensor(
            -1j * self.h_dense * self.config.time_step,
            dtype=torch.complex128,
            device=device,
        )
        self._exp_dt_gpu = torch.matrix_exp(h_gpu)
        self._initial_state_gpu = torch.tensor(
            self._initial_state, dtype=torch.complex128, device=device
        )
        logger.info(
            "Precomputed GPU matrix exponential: dim=%d", self.subspace_dim
        )

    def extract_projected_submatrix(
        self, configs: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract projected H and S from the precomputed dense Hamiltonian.

        Maps each configuration in *configs* to its index in the full
        particle-conserving subspace and extracts the corresponding
        submatrix of ``self.h_dense``.  Falls back to
        :func:`_build_projected_matrices` if any configuration is not
        found in the subspace.

        Parameters
        ----------
        configs : torch.Tensor
            Basis configurations, shape ``(n_basis, num_sites)``.

        Returns
        -------
        h_proj : np.ndarray
            Projected Hamiltonian, shape ``(n_basis, n_basis)``.
        s_proj : np.ndarray
            Overlap matrix (identity), shape ``(n_basis, n_basis)``.
        """
        if not self._subspace_hash_to_idx:
            return _build_projected_matrices(self.hamiltonian, configs)

        n = configs.shape[0]

        # Compute hashes for the requested configs
        if hasattr(self.hamiltonian, "_config_hash_batch"):
            hashes = self.hamiltonian._config_hash_batch(configs.cpu())
        else:
            num_sites = configs.shape[1]
            powers = torch.tensor(
                [1 << k for k in range(num_sites - 1, -1, -1)],
                dtype=torch.int64,
            )
            hashes = (
                configs.cpu().to(torch.int64) * powers.unsqueeze(0)
            ).sum(dim=-1)

        # Map each config to its subspace index
        indices = np.empty(n, dtype=np.intp)
        for i in range(n):
            h = int(hashes[i].item())
            idx = self._subspace_hash_to_idx.get(h)
            if idx is None:
                # Config not in precomputed subspace — fall back
                logger.debug(
                    "Config %d not in subspace index, falling back to full build", i
                )
                return _build_projected_matrices(self.hamiltonian, configs)
            indices[i] = idx

        h_proj = self.h_dense[np.ix_(indices, indices)].copy()
        s_proj = np.eye(n, dtype=np.float64)
        return h_proj, s_proj

    def _setup_particle_conserving_subspace(self) -> torch.Tensor:
        """Generate all valid particle-conserving configurations.

        For a molecular Hamiltonian with ``n_alpha`` spin-up and ``n_beta``
        spin-down electrons distributed across ``n_orb`` spatial orbitals
        (mapped to ``2 * n_orb`` qubits via Jordan--Wigner), enumerate all
        configurations that satisfy the particle-number constraints.

        Returns
        -------
        torch.Tensor
            All valid configurations, shape ``(n_configs, num_sites)``.
        """
        integrals = self.hamiltonian.integrals  # type: ignore[attr-defined]
        n_orb = integrals.n_orbitals
        n_alpha = integrals.n_alpha
        n_beta = integrals.n_beta
        num_qubits = 2 * n_orb

        configs: List[List[int]] = []

        # Alpha orbitals: qubits [0, n_orb)
        # Beta orbitals: qubits [n_orb, 2*n_orb)
        for alpha_occ in combinations(range(n_orb), n_alpha):
            for beta_occ in combinations(range(n_orb), n_beta):
                config = [0] * num_qubits
                for a in alpha_occ:
                    config[a] = 1
                for b in beta_occ:
                    config[n_orb + b] = 1
                configs.append(config)

        return torch.tensor(configs, dtype=torch.int64)

    def _compute_krylov_state(self, k: int) -> np.ndarray:
        r"""Compute the *k*-th Krylov state via matrix exponentiation.

        Evaluates :math:`|\psi_k\rangle = (e^{-iH\Delta t})^k |\psi_0\rangle`.

        When CUDA is available, uses a precomputed GPU matrix exponential
        ``U = e^{-iH\Delta t}`` and applies *k* matrix-vector products on
        the GPU.  Otherwise falls back to
        :func:`scipy.sparse.linalg.expm_multiply` on the CPU.

        Parameters
        ----------
        k : int
            Krylov power index (``k = 0`` returns the initial state).

        Returns
        -------
        np.ndarray
            State vector of shape ``(subspace_dim,)`` with dtype
            ``complex128``.
        """
        if k == 0:
            return self._initial_state.copy()

        # Lazy GPU precomputation
        if self._exp_dt_gpu is None and torch.cuda.is_available():
            self._precompute_gpu_time_evolution()

        # GPU path: repeated matrix-vector products
        if self._exp_dt_gpu is not None:
            state = self._initial_state_gpu.clone()
            for _ in range(k):
                state = self._exp_dt_gpu @ state
            return state.cpu().numpy()

        # CPU fallback: scipy expm_multiply
        total_time = k * self.config.time_step
        state = expm_multiply(
            -1j * self.h_dense,
            self._initial_state,
            start=0.0,
            stop=total_time,
            num=2,
            endpoint=True,
        )
        # expm_multiply returns shape (num, dim); take the last row
        return state[-1]

    def _sample_from_state(
        self, state_vector: np.ndarray, n_samples: int
    ) -> torch.Tensor:
        """Sample computational-basis configurations from a state vector.

        Parameters
        ----------
        state_vector : np.ndarray
            State vector of shape ``(subspace_dim,)`` with complex amplitudes.
        n_samples : int
            Number of configurations to sample.

        Returns
        -------
        torch.Tensor
            Unique sampled configurations, shape ``(n_unique, num_sites)``.
        """
        probabilities = np.abs(state_vector) ** 2
        prob_sum = probabilities.sum()
        if prob_sum < 1e-15:
            raise RuntimeError(
                "State vector has near-zero norm; cannot sample configurations."
            )
        probabilities = probabilities / prob_sum

        rng = np.random.default_rng()
        indices = rng.choice(len(probabilities), size=n_samples, p=probabilities)
        unique_indices = np.unique(indices)

        if self.subspace_configs is not None:
            sampled = self.subspace_configs[unique_indices]
        else:
            # Full Hilbert space: convert indices to binary configs
            num_sites = self.hamiltonian.num_sites
            configs_list = []
            for idx in unique_indices:
                config = self.hamiltonian._index_to_config(int(idx))
                configs_list.append(config)
            sampled = torch.stack(configs_list)

        return sampled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute the full SKQD algorithm.

        Constructs Krylov states iteratively, samples configurations from
        each, builds projected Hamiltonian and overlap matrices in the
        sampled basis, and solves the generalized eigenvalue problem.

        Returns
        -------
        eigenvalues : np.ndarray
            Lowest eigenvalues from the projected problem,
            shape ``(num_eigenvalues,)``.
        info : dict
            Diagnostic information with keys:

            - ``"basis_size"`` : int — final number of basis configurations.
            - ``"krylov_dim"`` : int — number of Krylov vectors constructed.
            - ``"energies_per_step"`` : list of float — ground-state energy
              estimate after each Krylov expansion step.
            - ``"basis_configs"`` : torch.Tensor — final basis configurations.
        """
        all_configs: Optional[torch.Tensor] = None
        energies_per_step: List[float] = []

        for k in range(self.config.max_krylov_dim):
            logger.info("Computing Krylov state k=%d", k)

            krylov_state = self._compute_krylov_state(k)
            sampled = self._sample_from_state(
                krylov_state, self.config.shots_per_krylov
            )

            if self.config.use_cumulative_basis and all_configs is not None:
                # Union of existing and new configs (deduplicated)
                combined = torch.cat([all_configs, sampled], dim=0)
                all_configs = torch.unique(combined, dim=0)
            else:
                all_configs = sampled

            # Build and solve projected problem (use submatrix extraction)
            h_proj, s_proj = self.extract_projected_submatrix(all_configs)
            eigenvalues, _ = _solve_generalised_eigenproblem(
                h_proj,
                s_proj,
                self.config.num_eigenvalues,
                self.config.regularization,
            )
            energies_per_step.append(float(eigenvalues[0]))
            logger.info(
                "Step k=%d: basis_size=%d, E0=%.10f",
                k,
                all_configs.shape[0],
                eigenvalues[0],
            )

        # Final solve
        assert all_configs is not None
        h_proj, s_proj = self.extract_projected_submatrix(all_configs)
        eigenvalues, eigenvectors = _solve_generalised_eigenproblem(
            h_proj,
            s_proj,
            self.config.num_eigenvalues,
            self.config.regularization,
        )

        info: Dict[str, Any] = {
            "basis_size": all_configs.shape[0],
            "krylov_dim": self.config.max_krylov_dim,
            "energies_per_step": energies_per_step,
            "basis_configs": all_configs,
        }

        return eigenvalues, info
