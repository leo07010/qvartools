"""
tfim — Transverse-field Ising model
=====================================

Provides ``TransverseFieldIsing``, the transverse-field Ising model with
tuneable interaction range on a one-dimensional chain with optional
periodic boundary conditions.

Configurations are binary vectors of length ``num_spins`` where
``0 ≡ spin-up (|↑⟩)`` and ``1 ≡ spin-down (|↓⟩)``.
"""

from __future__ import annotations

from typing import Tuple

import torch

from qvartools.hamiltonians.hamiltonian import Hamiltonian

__all__ = [
    "TransverseFieldIsing",
]


class TransverseFieldIsing(Hamiltonian):
    r"""Transverse-field Ising model with tuneable interaction range.

    .. math::

        H = -V \sum_{i} \sum_{l=1}^{L} S^z_i\, S^z_{i+l}
            - h \sum_i S^x_i

    where *L* controls the range of the ZZ interaction and the sum over
    *i* respects boundary conditions.

    Convention: ``config[i] = 0`` → spin-up (S^z = +½),
    ``config[i] = 1`` → spin-down (S^z = −½).

    Parameters
    ----------
    num_spins : int
        Number of spin sites.
    V : float, optional
        ZZ interaction strength (default ``1.0``).
    h : float, optional
        Transverse field strength (default ``1.0``).
    L : int, optional
        Range of the ZZ interaction in lattice units (default ``1``
        for nearest-neighbour).
    periodic : bool, optional
        Periodic boundary conditions (default ``True``).

    Examples
    --------
    >>> import torch
    >>> H = TransverseFieldIsing(num_spins=4, V=1.0, h=0.5)
    >>> config = torch.tensor([0, 1, 0, 1])
    >>> H.diagonal_element(config)
    tensor(...)
    >>> energy, state = H.exact_ground_state()
    """

    def __init__(
        self,
        num_spins: int,
        V: float = 1.0,
        h: float = 1.0,
        L: int = 1,
        periodic: bool = True,
    ) -> None:
        super().__init__(num_sites=num_spins, local_dim=2)

        self.V: float = float(V)
        self.h: float = float(h)
        self.L: int = int(L)
        self.periodic: bool = periodic

        # Precompute interaction pairs
        self._interaction_pairs: list[Tuple[int, int]] = []
        for i in range(num_spins):
            for ell in range(1, self.L + 1):
                j = i + ell
                if periodic:
                    j = j % num_spins
                elif j >= num_spins:
                    break
                self._interaction_pairs.append((i, j))

    # ------------------------------------------------------------------
    # Diagonal
    # ------------------------------------------------------------------

    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        """Compute ⟨config|H|config⟩ (ZZ interaction with range L).

        Parameters
        ----------
        config : torch.Tensor
            Shape ``(num_sites,)``.

        Returns
        -------
        torch.Tensor
            Scalar diagonal energy.
        """
        config_f = config.to(dtype=torch.float64)
        sz = 0.5 - config_f  # (num_sites,)

        energy = 0.0
        for i, j in self._interaction_pairs:
            energy -= self.V * float(sz[i].item()) * float(sz[j].item())

        return torch.tensor(energy, dtype=torch.float64, device=config.device)

    # ------------------------------------------------------------------
    # Off-diagonal
    # ------------------------------------------------------------------

    def get_connections(
        self, config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return off-diagonal connected states from the transverse field.

        The transverse field ``-h S^x_i`` flips each spin individually
        with amplitude ``-h/2``.

        Parameters
        ----------
        config : torch.Tensor
            Shape ``(num_sites,)``.

        Returns
        -------
        connected_configs : torch.Tensor
            Shape ``(n_conn, num_sites)``.
        matrix_elements : torch.Tensor
            Shape ``(n_conn,)``.
        """
        if abs(self.h) < 1e-15:
            return (
                torch.empty(
                    (0, self.num_sites), dtype=torch.int64, device=config.device
                ),
                torch.empty(0, dtype=torch.float64, device=config.device),
            )

        new_configs: list[torch.Tensor] = []
        coeff = -0.5 * self.h

        for site in range(self.num_sites):
            new_cfg = config.clone()
            new_cfg[site] = 1 - int(config[site].item())
            new_configs.append(new_cfg)

        return (
            torch.stack(new_configs).to(dtype=torch.int64, device=config.device),
            torch.full(
                (self.num_sites,), coeff, dtype=torch.float64, device=config.device
            ),
        )
