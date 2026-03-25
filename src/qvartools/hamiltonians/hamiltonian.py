"""
base — Abstract Hamiltonian base class
=======================================

Provides the ``Hamiltonian`` ABC that every concrete Hamiltonian must
implement, together with helper utilities for dense/sparse matrix
construction, exact diagonalisation, and configuration ↔ index mapping.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch

from qvartools.hamiltonians.pauli_string import PauliString  # noqa: F401

__all__ = [
    "Hamiltonian",
    "PauliString",
]


# ---------------------------------------------------------------------------
# Hamiltonian ABC
# ---------------------------------------------------------------------------


class Hamiltonian(ABC):
    """Abstract base class for all Hamiltonian operators.

    Every subclass must implement :meth:`diagonal_element` and
    :meth:`get_connections`.  The base class provides dense/sparse matrix
    construction, exact diagonalisation, and configuration bookkeeping.

    Parameters
    ----------
    num_sites : int
        Number of lattice / orbital sites.
    local_dim : int, optional
        Dimension of the local Hilbert space on each site (default ``2``
        for spin-½ / qubit).

    Attributes
    ----------
    num_sites : int
        Number of sites.
    local_dim : int
        Local Hilbert-space dimension.
    hilbert_dim : int
        Full Hilbert-space dimension ``local_dim ** num_sites``.

    Notes
    -----
    This is an abstract class and cannot be instantiated directly.
    Subclasses must implement :meth:`diagonal_element` and
    :meth:`get_connections`.

    Examples
    --------
    See :class:`~qvartools.hamiltonians.spin.heisenberg.HeisenbergHamiltonian`
    or :class:`~qvartools.hamiltonians.spin.tfim.TransverseFieldIsing` for
    concrete implementations.
    """

    def __init__(self, num_sites: int, local_dim: int = 2) -> None:
        if num_sites < 1:
            raise ValueError(f"num_sites must be >= 1, got {num_sites}")
        if local_dim < 2:
            raise ValueError(f"local_dim must be >= 2, got {local_dim}")
        self.num_sites: int = num_sites
        self.local_dim: int = local_dim
        self.hilbert_dim: int = local_dim ** num_sites

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        """Return the diagonal matrix element ⟨config|H|config⟩.

        Parameters
        ----------
        config : torch.Tensor
            Basis-state configuration, shape ``(num_sites,)`` with integer
            entries in ``[0, local_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar tensor (shape ``()``) with the diagonal element.
        """

    @abstractmethod
    def get_connections(
        self, config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return all states connected to *config* by the Hamiltonian.

        Parameters
        ----------
        config : torch.Tensor
            Basis-state configuration, shape ``(num_sites,)``.

        Returns
        -------
        connected_configs : torch.Tensor
            Off-diagonal connected configurations, shape ``(n_conn, num_sites)``.
        matrix_elements : torch.Tensor
            Corresponding matrix elements ⟨connected|H|config⟩,
            shape ``(n_conn,)``.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def matrix_element(
        self, config_i: torch.Tensor, config_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute a single matrix element ⟨config_i|H|config_j⟩.

        Parameters
        ----------
        config_i : torch.Tensor
            Bra configuration, shape ``(num_sites,)``.
        config_j : torch.Tensor
            Ket configuration, shape ``(num_sites,)``.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the matrix element.

        Examples
        --------
        >>> from qvartools.hamiltonians import TransverseFieldIsing
        >>> H = TransverseFieldIsing(num_spins=2, V=1.0, h=0.5)
        >>> import torch
        >>> H.matrix_element(torch.tensor([0, 0]), torch.tensor([0, 0]))
        tensor(...)
        """
        if torch.equal(config_i, config_j):
            return self.diagonal_element(config_j)

        connected, elements = self.get_connections(config_j)
        if connected.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=config_j.device)

        # Find config_i among the connected states
        matches = (connected == config_i.unsqueeze(0)).all(dim=-1)
        if matches.any():
            idx = matches.nonzero(as_tuple=False)[0, 0]
            return elements[idx]
        return torch.tensor(0.0, dtype=torch.float64, device=config_j.device)

    def matrix_elements(
        self, configs_bra: torch.Tensor, configs_ket: torch.Tensor
    ) -> torch.Tensor:
        """Compute the matrix H_{ij} = ⟨bra_i|H|ket_j⟩.

        Parameters
        ----------
        configs_bra : torch.Tensor
            Bra configurations, shape ``(M, num_sites)``.
        configs_ket : torch.Tensor
            Ket configurations, shape ``(N, num_sites)``.

        Returns
        -------
        torch.Tensor
            Matrix of shape ``(M, N)`` with dtype ``float64``.
        """
        m = configs_bra.shape[0]
        n = configs_ket.shape[0]
        h_matrix = torch.zeros(m, n, dtype=torch.float64, device=configs_bra.device)

        for j in range(n):
            ket = configs_ket[j]
            diag_val = self.diagonal_element(ket)
            connected, elements = self.get_connections(ket)

            for i in range(m):
                bra = configs_bra[i]
                if torch.equal(bra, ket):
                    h_matrix[i, j] = diag_val
                elif connected.numel() > 0:
                    matches = (connected == bra.unsqueeze(0)).all(dim=-1)
                    if matches.any():
                        idx = matches.nonzero(as_tuple=False)[0, 0]
                        h_matrix[i, j] = elements[idx]

        return h_matrix

    def to_dense(self, device: str = "cpu") -> torch.Tensor:
        """Build the full dense Hamiltonian matrix.

        Parameters
        ----------
        device : str, optional
            Torch device for the output tensor (default ``"cpu"``).

        Returns
        -------
        torch.Tensor
            Dense matrix of shape ``(hilbert_dim, hilbert_dim)``,
            dtype ``float64``.

        Warns
        -----
        UserWarning
            If ``num_sites > 16``, since the matrix may be very large.

        Examples
        --------
        >>> from qvartools.hamiltonians import TransverseFieldIsing
        >>> H = TransverseFieldIsing(num_spins=2, V=1.0, h=0.5)
        >>> H.to_dense().shape
        torch.Size([4, 4])
        """
        if self.num_sites > 16:
            warnings.warn(
                f"Building dense matrix for {self.num_sites} sites "
                f"(dim={self.hilbert_dim}). This may consume a lot of memory.",
                UserWarning,
                stacklevel=2,
            )

        configs = self._generate_all_configs(device=device)
        h_matrix = torch.zeros(
            self.hilbert_dim, self.hilbert_dim, dtype=torch.float64, device=device
        )

        for idx in range(self.hilbert_dim):
            config = configs[idx]
            h_matrix[idx, idx] = self.diagonal_element(config)

            connected, elements = self.get_connections(config)
            if connected.numel() == 0:
                continue

            for k in range(connected.shape[0]):
                col = self._config_to_index(connected[k])
                h_matrix[col, idx] = elements[k]

        return h_matrix

    def to_sparse(self, device: str = "cpu") -> "scipy.sparse.csr_matrix":  # noqa: F821
        """Build a SciPy CSR sparse Hamiltonian matrix.

        Parameters
        ----------
        device : str, optional
            Torch device used during construction (default ``"cpu"``).
            The returned sparse matrix always lives on the CPU (NumPy).

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of shape ``(hilbert_dim, hilbert_dim)``.
        """
        import scipy.sparse

        configs = self._generate_all_configs(device=device)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for idx in range(self.hilbert_dim):
            config = configs[idx]
            diag = self.diagonal_element(config)
            rows.append(idx)
            cols.append(idx)
            vals.append(float(diag))

            connected, elements = self.get_connections(config)
            if connected.numel() == 0:
                continue
            for k in range(connected.shape[0]):
                col = self._config_to_index(connected[k])
                rows.append(col)
                cols.append(idx)
                vals.append(float(elements[k]))

        return scipy.sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(self.hilbert_dim, self.hilbert_dim),
        )

    def exact_ground_state(
        self, device: str = "cpu"
    ) -> Tuple[float, torch.Tensor]:
        """Compute the exact ground state via full diagonalisation.

        Parameters
        ----------
        device : str, optional
            Torch device (default ``"cpu"``).

        Returns
        -------
        energy : float
            Ground-state energy (lowest eigenvalue).
        state : torch.Tensor
            Ground-state eigenvector, shape ``(hilbert_dim,)``.

        Examples
        --------
        >>> from qvartools.hamiltonians import TransverseFieldIsing
        >>> H = TransverseFieldIsing(num_spins=2, V=1.0, h=0.5)
        >>> energy, state = H.exact_ground_state()
        >>> isinstance(energy, float)
        True
        """
        h_dense = self.to_dense(device=device)
        eigenvalues, eigenvectors = torch.linalg.eigh(h_dense)
        energy = float(eigenvalues[0])
        state = eigenvectors[:, 0]
        return energy, state

    def ground_state_sparse(
        self, k: int = 1, device: str = "cpu"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the lowest *k* eigenstates via sparse diagonalisation.

        Uses :func:`scipy.sparse.linalg.eigsh` with ``which='SA'``.

        Parameters
        ----------
        k : int, optional
            Number of lowest eigenstates to compute (default ``1``).
        device : str, optional
            Torch device used during sparse matrix construction
            (default ``"cpu"``).

        Returns
        -------
        eigenvalues : np.ndarray
            Lowest *k* eigenvalues, shape ``(k,)``.
        eigenvectors : np.ndarray
            Corresponding eigenvectors, shape ``(hilbert_dim, k)``.
        """
        from scipy.sparse.linalg import eigsh

        h_sparse = self.to_sparse(device=device)
        eigenvalues, eigenvectors = eigsh(h_sparse, k=k, which="SA")
        order = np.argsort(eigenvalues)
        return eigenvalues[order], eigenvectors[:, order]

    # ------------------------------------------------------------------
    # Configuration utilities
    # ------------------------------------------------------------------

    def _generate_all_configs(self, device: str = "cpu") -> torch.Tensor:
        """Generate every computational-basis configuration.

        Parameters
        ----------
        device : str, optional
            Torch device (default ``"cpu"``).

        Returns
        -------
        torch.Tensor
            All configurations, shape ``(hilbert_dim, num_sites)``,
            dtype ``torch.int64``.  Each row is a basis state expressed
            in the local-dimension encoding (big-endian).
        """
        indices = torch.arange(self.hilbert_dim, device=device)
        configs = torch.zeros(
            self.hilbert_dim, self.num_sites, dtype=torch.int64, device=device
        )
        for site in range(self.num_sites - 1, -1, -1):
            configs[:, site] = indices % self.local_dim
            indices = indices // self.local_dim
        return configs

    def _config_to_index(self, config: torch.Tensor) -> int:
        """Convert a configuration tensor to its Hilbert-space index.

        Parameters
        ----------
        config : torch.Tensor
            Configuration vector, shape ``(num_sites,)``.

        Returns
        -------
        int
            Integer index in ``[0, hilbert_dim)``.
        """
        idx = 0
        for site in range(self.num_sites):
            idx = idx * self.local_dim + int(config[site].item())
        return idx

    def _index_to_config(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Convert a Hilbert-space index to its configuration tensor.

        Parameters
        ----------
        idx : int
            Integer index in ``[0, hilbert_dim)``.
        device : str, optional
            Torch device (default ``"cpu"``).

        Returns
        -------
        torch.Tensor
            Configuration vector, shape ``(num_sites,)``, dtype ``int64``.
        """
        config = torch.zeros(self.num_sites, dtype=torch.int64, device=device)
        remaining = idx
        for site in range(self.num_sites - 1, -1, -1):
            config[site] = remaining % self.local_dim
            remaining //= self.local_dim
        return config
