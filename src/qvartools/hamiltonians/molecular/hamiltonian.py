"""
molecular — MolecularHamiltonian with Jordan--Wigner mapping
=============================================================

Provides ``MolecularHamiltonian``, the second-quantised molecular
electronic Hamiltonian mapped to qubits via the Jordan--Wigner
transformation.  Supports both a single-configuration interface
(``get_connections``) and fully vectorised GPU-batch routines.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.hamiltonians.integrals import (
    MATRIX_ELEMENT_TOL,
    MolecularIntegrals,
    compute_molecular_integrals,
)
from qvartools.hamiltonians.molecular.jordan_wigner import _HAS_NUMBA
from qvartools.hamiltonians.molecular.slater_condon import numba_get_connections

__all__ = [
    "MolecularHamiltonian",
    # Re-export for backward compatibility
    "MolecularIntegrals",
    "compute_molecular_integrals",
]

logger = logging.getLogger(__name__)


# ===================================================================
# MolecularHamiltonian
# ===================================================================


class MolecularHamiltonian(Hamiltonian):
    """Second-quantised molecular Hamiltonian with Jordan--Wigner mapping.

    Spin-orbitals are ordered as ``[alpha_0, alpha_1, ..., alpha_{n-1},
    beta_0, beta_1, ..., beta_{n-1}]``.  A configuration is a binary
    occupation vector of length ``2 * n_orbitals``.

    Parameters
    ----------
    integrals : MolecularIntegrals
        Molecular integrals (one-body, two-body, metadata).
    device : str, optional
        Torch device for tensor storage (default ``"cpu"``).

    Attributes
    ----------
    integrals : MolecularIntegrals
        The stored integrals.
    n_orb : int
        Number of spatial orbitals.
    E_nuc : float
        Nuclear repulsion energy.
    device : str
        Torch device string.

    Examples
    --------
    >>> import numpy as np
    >>> from qvartools.hamiltonians import MolecularIntegrals, MolecularHamiltonian
    >>> h1 = np.diag([-1.0, -0.5])
    >>> h2 = np.zeros((2, 2, 2, 2))
    >>> mi = MolecularIntegrals(h1, h2, 0.7, 2, 2, 1, 1)
    >>> ham = MolecularHamiltonian(mi)
    >>> ham.num_sites
    4
    """

    def __init__(
        self,
        integrals: MolecularIntegrals,
        device: str = "cpu",
    ) -> None:
        n_orb = integrals.n_orbitals
        num_sites = 2 * n_orb
        super().__init__(num_sites=num_sites, local_dim=2)

        self.integrals = integrals
        self.n_orb = n_orb
        self.E_nuc = integrals.nuclear_repulsion
        self.device = device

        # Store integrals as float64 numpy arrays for Numba and as tensors
        self._h1e_np = np.ascontiguousarray(integrals.h1e, dtype=np.float64)
        self._h2e_np = np.ascontiguousarray(integrals.h2e, dtype=np.float64)

        # Torch tensors for vectorised operations
        self._h1e = torch.tensor(self._h1e_np, dtype=torch.float64, device=device)
        self._h2e = torch.tensor(self._h2e_np, dtype=torch.float64, device=device)

        # Precomputed diagonal quantities
        self._h1_diag = torch.tensor(
            np.diag(self._h1e_np).copy(), dtype=torch.float64, device=device
        )  # shape (n_orb,)

        # Coulomb tensor J[p,q] = h2e[p,p,q,q]
        self._J_tensor = torch.tensor(
            self._h2e_np[
                np.arange(n_orb)[:, None],
                np.arange(n_orb)[:, None],
                np.arange(n_orb)[None, :],
                np.arange(n_orb)[None, :],
            ],
            dtype=torch.float64,
            device=device,
        )  # shape (n_orb, n_orb)

        # Exchange tensor K[p,q] = h2e[p,q,q,p]
        self._K_tensor = torch.tensor(
            self._h2e_np[
                np.arange(n_orb)[:, None],
                np.arange(n_orb)[None, :],
                np.arange(n_orb)[None, :],
                np.arange(n_orb)[:, None],
            ],
            dtype=torch.float64,
            device=device,
        )  # shape (n_orb, n_orb)

        # Precomputed J_single[p,q,r] = h2e[p,q,r,r] and K_single[p,q,r] = h2e[p,r,r,q]
        r_idx = np.arange(n_orb)
        self._J_single_np = np.ascontiguousarray(
            self._h2e_np[:, :, r_idx, r_idx].transpose(0, 2, 1).copy()
            if n_orb > 0
            else np.empty((0, 0, 0), dtype=np.float64)
        )
        # Actually build properly
        J_single = np.zeros((n_orb, n_orb, n_orb), dtype=np.float64)
        K_single = np.zeros((n_orb, n_orb, n_orb), dtype=np.float64)
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    J_single[p, q, r] = self._h2e_np[p, q, r, r]
                    K_single[p, q, r] = self._h2e_np[p, r, r, q]
        self._J_single_np = np.ascontiguousarray(J_single)
        self._K_single_np = np.ascontiguousarray(K_single)

        # Precompute sparse h2e dictionaries for efficient double excitations
        self._h2e_sparse: Dict[Tuple[int, int, int, int], float] = {}
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        val = self._h2e_np[p, q, r, s]
                        if abs(val) > MATRIX_ELEMENT_TOL:
                            self._h2e_sparse[(p, q, r, s)] = val

        # Precompute excitation indices for vectorised batch operations
        self._precompute_excitation_indices()

        # Integer hashing helpers
        self._use_split_hash = num_sites >= 64

        logger.info(
            "MolecularHamiltonian initialised: %d orbitals, %d spin-orbitals, "
            "device=%s, numba=%s",
            n_orb,
            num_sites,
            device,
            _HAS_NUMBA,
        )

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_excitation_indices(self) -> None:
        """Precompute orbital-pair indices for vectorised excitations.

        Creates index tensors on ``self.device`` for the occupied→virtual
        single-excitation pairs and (occ, occ)→(virt, virt) double-excitation
        quadruplets, separated by spin sector.
        """
        n = self.n_orb
        # All alpha spin-orbital indices [0, n_orb)
        # All beta spin-orbital indices  [n_orb, 2*n_orb)
        self._alpha_range = torch.arange(0, n, device=self.device)
        self._beta_range = torch.arange(n, 2 * n, device=self.device)

    # ------------------------------------------------------------------
    # Integer hashing
    # ------------------------------------------------------------------

    def _config_hash(self, config: torch.Tensor) -> int:
        """Hash a binary configuration to an integer.

        For systems with fewer than 64 spin-orbitals the hash is a single
        Python ``int`` treating the config as a big-endian bitstring.
        For 64+ sites the config is split into two halves to avoid
        overflow in 64-bit integer arithmetic.

        Parameters
        ----------
        config : torch.Tensor
            Binary occupation vector, shape ``(num_sites,)``.

        Returns
        -------
        int
            Integer hash.
        """
        if self._use_split_hash:
            half = self.num_sites // 2
            hi = 0
            for i in range(half):
                hi = hi * 2 + int(config[i].item())
            lo = 0
            for i in range(half, self.num_sites):
                lo = lo * 2 + int(config[i].item())
            # Combine using a Cantor-like pairing
            return hi * (2 ** (self.num_sites - half)) + lo
        else:
            val = 0
            for i in range(self.num_sites):
                val = val * 2 + int(config[i].item())
            return val

    def _config_hash_batch(self, configs: torch.Tensor) -> torch.Tensor:
        """Hash a batch of configurations to integers.

        Uses integer bit-shifting (``1 << k``) instead of floating-point
        ``torch.pow`` to avoid GPU rounding errors in float64→int64
        conversion.

        Parameters
        ----------
        configs : torch.Tensor
            Binary occupation vectors, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Integer hashes, shape ``(batch,)``, dtype ``int64``.
        """
        powers = torch.tensor(
            [1 << k for k in range(self.num_sites - 1, -1, -1)],
            dtype=torch.int64,
            device=configs.device,
        )
        return (configs.to(torch.int64) * powers.unsqueeze(0)).sum(dim=-1)

    # ------------------------------------------------------------------
    # Diagonal elements
    # ------------------------------------------------------------------

    def diagonal_elements_batch(self, configs: torch.Tensor) -> torch.Tensor:
        """Compute diagonal matrix elements for a batch of configurations.

        Uses vectorised ``einsum`` operations for the one-body, Coulomb,
        and exchange contributions.

        Parameters
        ----------
        configs : torch.Tensor
            Occupation vectors, shape ``(batch, num_sites)``,
            dtype ``int64`` or ``float64``.

        Returns
        -------
        torch.Tensor
            Diagonal elements, shape ``(batch,)``, dtype ``float64``.
        """
        configs_f = configs.to(dtype=torch.float64, device=self.device)
        batch = configs_f.shape[0]
        n = self.n_orb

        # Split into alpha and beta occupations (spatial orbital basis)
        occ_alpha = configs_f[:, :n]   # (batch, n_orb)
        occ_beta = configs_f[:, n:]    # (batch, n_orb)

        # One-body energy: sum_p h_pp * (n_alpha_p + n_beta_p)
        occ_total = occ_alpha + occ_beta  # (batch, n_orb)
        e_one = torch.einsum("bp,p->b", occ_total, self._h1_diag)

        # Coulomb: 0.5 * sum_{pq} J[p,q] * n_p * n_q  (over all spin-orbitals)
        # J[p,q] = h2e[p,p,q,q]
        # n_p comes from both alpha and beta
        e_coulomb = 0.5 * torch.einsum(
            "bp,pq,bq->b", occ_total, self._J_tensor, occ_total
        )

        # Exchange: -0.5 * sum_{pq} K[p,q] * n_p^sigma * n_q^sigma (same spin)
        # K[p,q] = h2e[p,q,q,p]
        e_exchange_alpha = -0.5 * torch.einsum(
            "bp,pq,bq->b", occ_alpha, self._K_tensor, occ_alpha
        )
        e_exchange_beta = -0.5 * torch.einsum(
            "bp,pq,bq->b", occ_beta, self._K_tensor, occ_beta
        )

        return self.E_nuc + e_one + e_coulomb + e_exchange_alpha + e_exchange_beta

    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        """Compute the diagonal element ⟨config|H|config⟩.

        Parameters
        ----------
        config : torch.Tensor
            Occupation vector, shape ``(num_sites,)``.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the diagonal energy.
        """
        return self.diagonal_elements_batch(config.unsqueeze(0)).squeeze(0)

    # ------------------------------------------------------------------
    # Off-diagonal connections (single config)
    # ------------------------------------------------------------------

    def get_connections(
        self, config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return all off-diagonal connected states via Slater--Condon rules.

        Computes single and double excitations with proper Jordan--Wigner
        signs.  Uses Numba kernels when available, otherwise falls back to
        a pure-Python implementation.

        Parameters
        ----------
        config : torch.Tensor
            Occupation vector, shape ``(num_sites,)``.

        Returns
        -------
        connected_configs : torch.Tensor
            Shape ``(n_conn, num_sites)``, dtype ``int64``.
        matrix_elements : torch.Tensor
            Shape ``(n_conn,)``, dtype ``float64``.
        """
        config_np = config.detach().cpu().numpy().astype(np.int64)

        if _HAS_NUMBA:
            conn_np, elem_np = numba_get_connections(
                config_np,
                self.n_orb,
                self._J_single_np,
                self._K_single_np,
                self._h1e_np,
                self._h2e_np,
                self.num_sites,
            )
        else:
            conn_np, elem_np = self._python_get_connections(config_np)

        # Filter by tolerance
        mask = np.abs(elem_np) > MATRIX_ELEMENT_TOL
        conn_np = conn_np[mask]
        elem_np = elem_np[mask]

        if conn_np.shape[0] == 0:
            return (
                torch.empty((0, self.num_sites), dtype=torch.int64, device=config.device),
                torch.empty(0, dtype=torch.float64, device=config.device),
            )

        return (
            torch.tensor(conn_np, dtype=torch.int64, device=config.device),
            torch.tensor(elem_np, dtype=torch.float64, device=config.device),
        )

    def _python_get_connections(
        self, config: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pure-Python fallback for :meth:`get_connections`.

        Parameters
        ----------
        config : np.ndarray
            Occupation vector, shape ``(num_sites,)``.

        Returns
        -------
        configs : np.ndarray
            Connected configurations, shape ``(n_conn, num_sites)``.
        elements : np.ndarray
            Matrix elements, shape ``(n_conn,)``.
        """
        all_configs: list[np.ndarray] = []
        all_elements: list[float] = []

        n_orb = self.n_orb
        num_sites = self.num_sites

        # ---------- Single excitations ----------
        for q_spin in range(num_sites):
            if config[q_spin] == 0:
                continue
            q_spatial = q_spin % n_orb
            q_is_alpha = q_spin < n_orb

            for p_spin in range(num_sites):
                if config[p_spin] == 1:
                    continue
                p_spatial = p_spin % n_orb
                p_is_alpha = p_spin < n_orb

                if p_is_alpha != q_is_alpha:
                    continue

                h_pq = self._h1e_np[p_spatial, q_spatial]

                two_body = 0.0
                for r_spin in range(num_sites):
                    if r_spin == q_spin or config[r_spin] == 0:
                        continue
                    r_spatial = r_spin % n_orb
                    r_is_alpha = r_spin < n_orb

                    two_body += self._h2e_np[p_spatial, q_spatial, r_spatial, r_spatial]
                    if p_is_alpha == r_is_alpha:
                        two_body -= self._h2e_np[p_spatial, r_spatial, r_spatial, q_spatial]

                me = h_pq + two_body
                if abs(me) < MATRIX_ELEMENT_TOL:
                    continue

                sign = self._jw_sign_single_py(config, p_spin, q_spin)
                new_cfg = config.copy()
                new_cfg[q_spin] = 0
                new_cfg[p_spin] = 1
                all_configs.append(new_cfg)
                all_elements.append(sign * me)

        # ---------- Double excitations ----------
        for q_spin in range(num_sites):
            if config[q_spin] == 0:
                continue
            q_spatial = q_spin % n_orb
            q_is_alpha = q_spin < n_orb

            for s_spin in range(q_spin + 1, num_sites):
                if config[s_spin] == 0:
                    continue
                s_spatial = s_spin % n_orb
                s_is_alpha = s_spin < n_orb

                for p_spin in range(num_sites):
                    if config[p_spin] == 1:
                        continue
                    p_spatial = p_spin % n_orb
                    p_is_alpha = p_spin < n_orb

                    for r_spin in range(p_spin + 1, num_sites):
                        if config[r_spin] == 1:
                            continue
                        r_spatial = r_spin % n_orb
                        r_is_alpha = r_spin < n_orb

                        spin_in = int(q_is_alpha) + int(s_is_alpha)
                        spin_out = int(p_is_alpha) + int(r_is_alpha)
                        if spin_in != spin_out:
                            continue

                        if p_is_alpha == r_is_alpha:
                            if p_is_alpha == q_is_alpha and r_is_alpha == s_is_alpha:
                                me = (
                                    self._h2e_np[p_spatial, q_spatial, r_spatial, s_spatial]
                                    - self._h2e_np[p_spatial, s_spatial, r_spatial, q_spatial]
                                )
                            elif p_is_alpha == s_is_alpha and r_is_alpha == q_is_alpha:
                                me = (
                                    self._h2e_np[p_spatial, s_spatial, r_spatial, q_spatial]
                                    - self._h2e_np[p_spatial, q_spatial, r_spatial, s_spatial]
                                )
                            else:
                                continue
                        else:
                            if p_is_alpha == q_is_alpha and r_is_alpha == s_is_alpha:
                                me = self._h2e_np[p_spatial, q_spatial, r_spatial, s_spatial]
                            elif p_is_alpha == s_is_alpha and r_is_alpha == q_is_alpha:
                                me = self._h2e_np[p_spatial, s_spatial, r_spatial, q_spatial]
                            else:
                                continue

                        if abs(me) < MATRIX_ELEMENT_TOL:
                            continue

                        sign = self._jw_sign_double_py(config, p_spin, r_spin, q_spin, s_spin)
                        new_cfg = config.copy()
                        new_cfg[q_spin] = 0
                        new_cfg[s_spin] = 0
                        new_cfg[p_spin] = 1
                        new_cfg[r_spin] = 1
                        all_configs.append(new_cfg)
                        all_elements.append(sign * me)

        if not all_configs:
            return (
                np.empty((0, num_sites), dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        return np.array(all_configs, dtype=np.int64), np.array(all_elements, dtype=np.float64)

    @staticmethod
    def _jw_sign_single_py(config: np.ndarray, p: int, q: int) -> int:
        """Pure-Python Jordan--Wigner sign for single excitation.

        Parameters
        ----------
        config : np.ndarray
            Occupation vector.
        p : int
            Creation orbital.
        q : int
            Annihilation orbital.

        Returns
        -------
        int
            ``+1`` or ``-1``.
        """
        lo = min(p, q) + 1
        hi = max(p, q)
        count = int(np.sum(config[lo:hi]))
        return 1 - 2 * (count % 2)

    @staticmethod
    def _jw_sign_double_py(
        config: np.ndarray, p: int, r: int, q: int, s: int
    ) -> int:
        """Pure-Python Jordan--Wigner sign for double excitation.

        Operator ordering: a†_p a†_r a_s a_q (right-to-left).

        Parameters
        ----------
        config : np.ndarray
            Occupation vector.
        p : int
            First creation orbital.
        r : int
            Second creation orbital.
        q : int
            First annihilation orbital.
        s : int
            Second annihilation orbital.

        Returns
        -------
        int
            ``+1`` or ``-1``.
        """
        state = config.copy()
        sign = 1

        # Annihilate q
        count_q = int(np.sum(state[:q]))
        sign *= 1 - 2 * (count_q % 2)
        state[q] = 0

        # Annihilate s
        count_s = int(np.sum(state[:s]))
        sign *= 1 - 2 * (count_s % 2)
        state[s] = 0

        # Create r
        count_r = int(np.sum(state[:r]))
        sign *= 1 - 2 * (count_r % 2)
        state[r] = 1

        # Create p
        count_p = int(np.sum(state[:p]))
        sign *= 1 - 2 * (count_p % 2)
        state[p] = 1

        return sign

    # ------------------------------------------------------------------
    # Vectorised batch connections
    # ------------------------------------------------------------------

    def get_connections_vectorized_batch(
        self, configs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute connections for a batch of configs (GPU-vectorised).

        This method processes each configuration independently but uses
        GPU tensors throughout to avoid CPU↔GPU transfers within each
        configuration's excitation enumeration.

        Parameters
        ----------
        configs : torch.Tensor
            Occupation vectors, shape ``(batch, num_sites)``.

        Returns
        -------
        all_connected : list of torch.Tensor
            One tensor per batch element, each of shape
            ``(n_conn_i, num_sites)``.
        all_elements : list of torch.Tensor
            One tensor per batch element, each of shape ``(n_conn_i,)``.
        """
        batch = configs.shape[0]
        all_connected: List[torch.Tensor] = []
        all_elements: List[torch.Tensor] = []

        for b in range(batch):
            conn, elem = self.get_connections(configs[b])
            all_connected.append(conn)
            all_elements.append(elem)

        return all_connected, all_elements

    # ------------------------------------------------------------------
    # Optimised matrix_elements via hashing
    # ------------------------------------------------------------------

    def matrix_elements(
        self, configs_bra: torch.Tensor, configs_ket: torch.Tensor
    ) -> torch.Tensor:
        """Compute H_{ij} = ⟨bra_i|H|ket_j⟩ using hash-based lookup.

        This override of the base-class method uses integer hashing to
        avoid the O(n_conn * M) inner loop for matching bra states.
        Uses vectorised batch hashing for both bra and connected configs.

        Parameters
        ----------
        configs_bra : torch.Tensor
            Bra configurations, shape ``(M, num_sites)``.
        configs_ket : torch.Tensor
            Ket configurations, shape ``(N, num_sites)``.

        Returns
        -------
        torch.Tensor
            Matrix of shape ``(M, N)``, dtype ``float64``.
        """
        m = configs_bra.shape[0]
        n = configs_ket.shape[0]
        device = configs_bra.device

        # If bra == ket (same object), delegate to the symmetric fast path
        if configs_bra is configs_ket or (
            configs_bra.shape == configs_ket.shape
            and configs_bra.data_ptr() == configs_ket.data_ptr()
        ):
            return self.matrix_elements_fast(configs_bra)

        h_matrix = torch.zeros(m, n, dtype=torch.float64, device=device)

        # Build sorted bra hashes for vectorised searchsorted matching
        bra_hashes = self._config_hash_batch(configs_bra)  # (M,)
        sorted_bra_hashes, bra_sort_perm = torch.sort(bra_hashes)

        # Batch diagonal elements for all kets
        diag_vals = self.diagonal_elements_batch(configs_ket)

        # Batch hash all kets for diagonal matching
        ket_hashes = self._config_hash_batch(configs_ket)  # (N,)

        # Vectorised diagonal matching
        diag_positions = torch.searchsorted(
            sorted_bra_hashes.cpu(), ket_hashes.cpu()
        ).clamp(max=m - 1)
        diag_matched = sorted_bra_hashes.cpu()[diag_positions] == ket_hashes.cpu()
        if diag_matched.any():
            matched_j = torch.where(diag_matched)[0]
            matched_i = bra_sort_perm.cpu()[diag_positions[diag_matched]]
            h_matrix[matched_i.to(device), matched_j.to(device)] = diag_vals[matched_j.to(device)]

        # Pre-move kets to CPU once for Numba calls
        configs_ket_cpu = configs_ket.detach().cpu()

        for j in range(n):
            # Off-diagonal connections
            connected, elements = self.get_connections(configs_ket_cpu[j])
            if connected.numel() == 0:
                continue

            # Vectorised membership test via searchsorted
            conn_hashes = self._config_hash_batch(connected)  # (n_conn,)
            positions = torch.searchsorted(
                sorted_bra_hashes.cpu(), conn_hashes
            ).clamp(max=m - 1)
            matched_mask = sorted_bra_hashes.cpu()[positions] == conn_hashes
            if not matched_mask.any():
                continue

            orig_indices = bra_sort_perm.cpu()[positions[matched_mask]]
            matched_vals = elements[matched_mask]
            h_matrix[
                orig_indices.to(device),
                j,
            ] = matched_vals.to(dtype=torch.float64, device=device)

        return h_matrix

    # ------------------------------------------------------------------
    # Fast symmetric matrix construction
    # ------------------------------------------------------------------

    def matrix_elements_fast(
        self, configs: torch.Tensor
    ) -> torch.Tensor:
        """Build a Hermitian projected Hamiltonian matrix efficiently.

        Builds only the lower triangle via hash-based matching then
        mirrors to the upper triangle, guaranteeing exact symmetry.

        Uses fully vectorised hash lookups: ``torch.searchsorted`` on
        sorted basis hashes replaces the per-connection Python loop,
        giving O(n_conn * log(n_configs)) matching per ket instead of
        O(n_conn) Python dict lookups.

        Parameters
        ----------
        configs : torch.Tensor
            Basis configurations, shape ``(n_configs, num_sites)``.

        Returns
        -------
        torch.Tensor
            Hermitian matrix of shape ``(n_configs, n_configs)``,
            dtype ``float64``.

        Raises
        ------
        MemoryError
            If ``n_configs > 10000`` (dense matrix would be too large).
        """
        configs = configs.to(self.device)
        n_configs = configs.shape[0]

        if n_configs > 10000:
            mem_gb = n_configs ** 2 * 8 / 1e9
            raise MemoryError(
                f"matrix_elements_fast() refused {n_configs}x{n_configs} "
                f"dense matrix ({mem_gb:.1f} GB). Use sparse methods for "
                f"systems with >10000 configs."
            )

        H = torch.zeros(
            n_configs, n_configs, dtype=torch.float64, device=self.device
        )

        # Vectorised diagonal (already GPU-optimised)
        H.diagonal().copy_(self.diagonal_elements_batch(configs))

        # Vectorised batch hashing + sorting for searchsorted matching
        basis_hashes = self._config_hash_batch(configs)  # (n_configs,)
        sorted_hashes, sort_perm = torch.sort(basis_hashes)

        # Pre-move configs to CPU once for Numba calls
        configs_cpu = configs.detach().cpu()

        # Off-diagonal via get_connections for each ket
        for j in range(n_configs):
            connected, elements = self.get_connections(configs_cpu[j])
            if connected.numel() == 0:
                continue

            # Batch hash all connected configs at once
            conn_hashes = self._config_hash_batch(connected)  # (n_conn,) on CPU

            # Vectorised membership test via searchsorted on sorted hashes
            positions = torch.searchsorted(
                sorted_hashes.cpu(), conn_hashes
            )
            positions = positions.clamp(max=n_configs - 1)

            # Check which positions actually match
            matched_mask = sorted_hashes.cpu()[positions] == conn_hashes
            if not matched_mask.any():
                continue

            # Map back from sorted to original indices
            valid_pos = positions[matched_mask]
            orig_indices = sort_perm.cpu()[valid_pos]

            # Filter out self-matches (i != j)
            not_self = orig_indices != j
            if not not_self.any():
                continue

            final_i = orig_indices[not_self].to(self.device)
            final_vals = elements[matched_mask][not_self].to(
                dtype=torch.float64, device=self.device
            )

            H[final_i, j] = final_vals
            H[j, final_i] = final_vals

        return H

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @property
    def n_orbitals(self) -> int:
        """Number of spatial orbitals."""
        return self.n_orb

    @property
    def n_alpha(self) -> int:
        """Number of alpha electrons."""
        return self.integrals.n_alpha

    @property
    def n_beta(self) -> int:
        """Number of beta electrons."""
        return self.integrals.n_beta

    @property
    def h1e(self) -> torch.Tensor:
        """One-electron integral tensor."""
        return self._h1e

    @property
    def h2e(self) -> torch.Tensor:
        """Two-electron integral tensor."""
        return self._h2e

    def get_hf_state(self) -> torch.Tensor:
        """Return the Hartree--Fock ground-state configuration.

        Alpha electrons fill the lowest alpha spin-orbitals, beta
        electrons fill the lowest beta spin-orbitals.

        Returns
        -------
        torch.Tensor
            Binary occupation vector, shape ``(num_sites,)``, dtype ``int64``.

        Examples
        --------
        >>> import numpy as np
        >>> from qvartools.hamiltonians import MolecularIntegrals, MolecularHamiltonian
        >>> mi = MolecularIntegrals(np.eye(2), np.zeros((2,2,2,2)), 0.0, 2, 2, 1, 1)
        >>> ham = MolecularHamiltonian(mi)
        >>> ham.get_hf_state()
        tensor([1, 0, 0, 1])
        """
        config = torch.zeros(self.num_sites, dtype=torch.int64, device=self.device)
        config[: self.integrals.n_alpha] = 1
        config[self.n_orb : self.n_orb + self.integrals.n_beta] = 1
        return config

    def fci_energy(self) -> float:
        """Compute the Full-CI energy using PySCF's native FCI solver.

        Uses PySCF's Davidson-iteration FCI solver in the compressed
        alpha/beta string representation, which is vastly faster than
        building the full 2^n dense matrix.  Falls back to the GPU FCI
        solver if CuPy is available, or to brute-force dense
        diagonalisation for very small systems (≤ 8 qubits).

        Returns
        -------
        float
            The ground-state energy in Hartree.

        Raises
        ------
        RuntimeError
            If PySCF is not installed and ``num_sites > 20`` (dense
            diagonalisation would be intractable).
        """
        # Try PySCF's native FCI solver first (fastest)
        try:
            from pyscf import fci as pyscf_fci
            from pyscf import ao2mo

            h1e_np = self._h1e_np
            h2e_np = self._h2e_np
            norb = self.n_orb
            nelec = (self.integrals.n_alpha, self.integrals.n_beta)

            # Convert 4-index h2e to compressed 2-index form for PySCF
            h2e_flat = h2e_np.reshape(norb * norb, norb * norb)
            eri = ao2mo.restore(4, h2e_flat, norb)

            cisolver = pyscf_fci.direct_spin1.FCI()
            cisolver.conv_tol = 1e-12
            cisolver.max_cycle = 300
            cisolver.verbose = 0

            e_corr, _civec = cisolver.kernel(h1e_np, eri, norb, nelec)
            total_energy = float(e_corr) + self.E_nuc

            logger.info(
                "FCI energy (PySCF native): %.12f Ha", total_energy
            )
            return total_energy

        except ImportError:
            pass

        # Fallback: brute-force dense diag for small systems
        if self.num_sites > 20:
            raise RuntimeError(
                f"FCI with {self.num_sites} spin-orbitals "
                f"(dim={self.hilbert_dim}) is too large for dense diag. "
                "Install PySCF for efficient FCI: pip install pyscf"
            )
        energy, _ = self.exact_ground_state(device="cpu")
        return energy
