"""
slater_condon — Slater--Condon excitation kernels
==================================================

Provides Numba-accelerated (with pure-Python fallback) functions for
enumerating single and double excitations from a given occupation
configuration using Slater--Condon rules together with Jordan--Wigner
sign factors.

Functions
---------
_numba_single_excitations
    All single excitations from a configuration.
_numba_double_excitations
    All double excitations from a configuration.
numba_get_connections
    Combined single + double excitations.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from qvartools.hamiltonians.molecular.jordan_wigner import (
    njit,
    numba_jw_sign_double,
    numba_jw_sign_single,
)

__all__ = [
    "_numba_single_excitations",
    "_numba_double_excitations",
    "numba_get_connections",
]


@njit(cache=True)
def _numba_single_excitations(
    config: np.ndarray,
    n_orb: int,
    J_single: np.ndarray,
    K_single: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    num_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute all single excitations from *config* with Slater--Condon rules.

    Parameters
    ----------
    config : np.ndarray
        Occupation vector, shape ``(num_sites,)``.
    n_orb : int
        Number of spatial orbitals.
    J_single : np.ndarray
        Precomputed Coulomb integrals for singles, shape ``(n_orb, n_orb, n_orb)``.
        ``J_single[p, q, r] = h2e[p, q, r, r]``.
    K_single : np.ndarray
        Precomputed exchange integrals for singles, shape ``(n_orb, n_orb, n_orb)``.
        ``K_single[p, q, r] = h2e[p, r, r, q]``.
    h1e : np.ndarray
        One-electron integrals, shape ``(n_orb, n_orb)``.
    h2e : np.ndarray
        Two-electron integrals, shape ``(n_orb, n_orb, n_orb, n_orb)``.
    num_sites : int
        Total number of spin-orbitals (``2 * n_orb``).

    Returns
    -------
    new_configs : np.ndarray
        Connected configurations, shape ``(n_conn, num_sites)``.
    elements : np.ndarray
        Matrix elements, shape ``(n_conn,)``.

    Notes
    -----
    For a single excitation :math:`a^\\dagger_{p\\sigma} a_{q\\sigma}`, the
    Slater--Condon matrix element is

    .. math::

        \\langle \\Phi' | H | \\Phi \rangle
        = h_{pq} + \\sum_{r \\in \text{occ}} \bigl[
            (pq|rr) - \\delta_{\\sigma\\sigma_r} (pr|rq)
          \bigr]

    where the sum runs over all occupied spin-orbitals *r* other than *q*.
    """
    max_conn = num_sites * num_sites
    new_configs = np.empty((max_conn, num_sites), dtype=np.int64)
    elements = np.empty(max_conn, dtype=np.float64)
    count = 0

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

            # Spin conservation
            if p_is_alpha != q_is_alpha:
                continue

            # One-body integral
            h_pq = h1e[p_spatial, q_spatial]

            # Two-body contribution: sum over occupied r
            two_body = 0.0
            for r_spin in range(num_sites):
                if r_spin == q_spin:
                    continue
                if config[r_spin] == 0:
                    continue
                r_spatial = r_spin % n_orb
                r_is_alpha = r_spin < n_orb

                # Coulomb: always present
                two_body += h2e[p_spatial, q_spatial, r_spatial, r_spatial]

                # Exchange: only for same spin
                if p_is_alpha == r_is_alpha:
                    two_body -= h2e[p_spatial, r_spatial, r_spatial, q_spatial]

            me = h_pq + two_body

            if abs(me) < 1e-12:
                continue

            # JW sign
            sign = numba_jw_sign_single(config, p_spin, q_spin)

            new_cfg = config.copy()
            new_cfg[q_spin] = 0
            new_cfg[p_spin] = 1

            new_configs[count] = new_cfg
            elements[count] = sign * me
            count += 1

    return new_configs[:count].copy(), elements[:count].copy()


@njit(cache=True)
def _numba_double_excitations(
    config: np.ndarray,
    n_orb: int,
    h2e: np.ndarray,
    num_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute all double excitations from *config* with Slater--Condon rules.

    Parameters
    ----------
    config : np.ndarray
        Occupation vector, shape ``(num_sites,)``.
    n_orb : int
        Number of spatial orbitals.
    h2e : np.ndarray
        Two-electron integrals, shape ``(n_orb, n_orb, n_orb, n_orb)``.
    num_sites : int
        Total number of spin-orbitals.

    Returns
    -------
    new_configs : np.ndarray
        Connected configurations, shape ``(n_conn, num_sites)``.
    elements : np.ndarray
        Matrix elements, shape ``(n_conn,)``.

    Notes
    -----
    For a double excitation, the Slater--Condon rule gives

    .. math::

        \\langle \\Phi' | H | \\Phi \rangle
        = (pr|qs) - \\delta_{\\sigma_p \\sigma_r} (ps|qr)

    where the anti-symmetrised form applies when both created
    particles share the same spin.
    """
    max_conn = num_sites * num_sites * num_sites * num_sites
    # Use a more reasonable upper bound
    max_conn = min(max_conn, num_sites * num_sites * num_sites)
    new_configs = np.empty((max_conn, num_sites), dtype=np.int64)
    elements = np.empty(max_conn, dtype=np.float64)
    count = 0

    # Iterate over occupied pairs (q < s) annihilation
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

            # Iterate over virtual pairs (p < r) creation
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

                    # Spin conservation: total spin in == total spin out
                    spin_in = (1 if q_is_alpha else 0) + (1 if s_is_alpha else 0)
                    spin_out = (1 if p_is_alpha else 0) + (1 if r_is_alpha else 0)
                    if spin_in != spin_out:
                        continue

                    # Determine matrix element based on spin cases
                    # alpha-alpha or beta-beta
                    if p_is_alpha == r_is_alpha:
                        # Both same spin — include exchange
                        # Need to match creation/annihilation pairs by spin
                        if p_is_alpha == q_is_alpha and r_is_alpha == s_is_alpha:
                            me = (
                                h2e[p_spatial, q_spatial, r_spatial, s_spatial]
                                - h2e[p_spatial, s_spatial, r_spatial, q_spatial]
                            )
                        elif p_is_alpha == s_is_alpha and r_is_alpha == q_is_alpha:
                            me = (
                                h2e[p_spatial, s_spatial, r_spatial, q_spatial]
                                - h2e[p_spatial, q_spatial, r_spatial, s_spatial]
                            )
                        else:
                            continue
                    else:
                        # alpha-beta: no exchange
                        if p_is_alpha == q_is_alpha and r_is_alpha == s_is_alpha:
                            me = h2e[p_spatial, q_spatial, r_spatial, s_spatial]
                        elif p_is_alpha == s_is_alpha and r_is_alpha == q_is_alpha:
                            me = h2e[p_spatial, s_spatial, r_spatial, q_spatial]
                        else:
                            continue

                    if abs(me) < 1e-12:
                        continue

                    sign = numba_jw_sign_double(config, p_spin, r_spin, q_spin, s_spin)

                    new_cfg = config.copy()
                    new_cfg[q_spin] = 0
                    new_cfg[s_spin] = 0
                    new_cfg[p_spin] = 1
                    new_cfg[r_spin] = 1

                    if count >= max_conn:
                        break
                    new_configs[count] = new_cfg
                    elements[count] = sign * me
                    count += 1

    return new_configs[:count].copy(), elements[:count].copy()


@njit(cache=True)
def numba_get_connections(
    config: np.ndarray,
    n_orb: int,
    J_single: np.ndarray,
    K_single: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    num_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute all single and double excitations via Numba.

    Parameters
    ----------
    config : np.ndarray
        Occupation vector, shape ``(num_sites,)``.
    n_orb : int
        Number of spatial orbitals.
    J_single : np.ndarray
        Coulomb integrals for singles, shape ``(n_orb, n_orb, n_orb)``.
    K_single : np.ndarray
        Exchange integrals for singles, shape ``(n_orb, n_orb, n_orb)``.
    h1e : np.ndarray
        One-electron integrals, shape ``(n_orb, n_orb)``.
    h2e : np.ndarray
        Two-electron integrals, shape ``(n_orb, n_orb, n_orb, n_orb)``.
    num_sites : int
        Total number of spin-orbitals.

    Returns
    -------
    configs : np.ndarray
        Connected configurations, shape ``(n_conn, num_sites)``.
    elements : np.ndarray
        Matrix elements, shape ``(n_conn,)``.
    """
    s_configs, s_elements = _numba_single_excitations(
        config, n_orb, J_single, K_single, h1e, h2e, num_sites
    )
    d_configs, d_elements = _numba_double_excitations(
        config, n_orb, h2e, num_sites
    )

    n_single = s_configs.shape[0]
    n_double = d_configs.shape[0]
    total = n_single + n_double

    if total == 0:
        return np.empty((0, num_sites), dtype=np.int64), np.empty(0, dtype=np.float64)

    all_configs = np.empty((total, num_sites), dtype=np.int64)
    all_elements = np.empty(total, dtype=np.float64)

    for i in range(n_single):
        all_configs[i] = s_configs[i]
        all_elements[i] = s_elements[i]
    for i in range(n_double):
        all_configs[n_single + i] = d_configs[i]
        all_elements[n_single + i] = d_elements[i]

    return all_configs, all_elements
