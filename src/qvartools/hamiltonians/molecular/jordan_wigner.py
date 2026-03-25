"""
jordan_wigner — Jordan--Wigner sign computation kernels
========================================================

Provides Numba-accelerated (with pure-Python fallback) functions for
computing the fermionic sign factors arising from the Jordan--Wigner
transformation for single and double excitations.

The module also exports the ``_HAS_NUMBA`` flag and the ``njit`` shim
so that downstream modules can decorate their own kernels consistently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "_HAS_NUMBA",
    "njit",
    "numba_jw_sign_single",
    "numba_jw_sign_double",
]

# ---------------------------------------------------------------------------
# Optional Numba import
# ---------------------------------------------------------------------------

try:
    import numba  # noqa: F401
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    # Provide a no-op decorator so the function bodies below parse without
    # error even when numba is absent.
    def njit(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """No-op replacement for ``numba.njit`` when Numba is not installed."""
        def decorator(fn: Any) -> Any:
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator


# ---------------------------------------------------------------------------
# JW sign functions
# ---------------------------------------------------------------------------


@njit(cache=True)
def numba_jw_sign_single(
    config: np.ndarray, p: int, q: int
) -> int:
    """Compute the Jordan--Wigner sign for a single excitation a†_p a_q.

    The sign is ``(-1)^(number of occupied orbitals strictly between p and q)``.

    Parameters
    ----------
    config : np.ndarray
        Occupation vector, shape ``(num_sites,)``, dtype int.
    p : int
        Creation orbital index.
    q : int
        Annihilation orbital index.

    Returns
    -------
    int
        ``+1`` or ``-1``.

    Notes
    -----
    The Jordan--Wigner sign arises from anti-commuting the fermionic
    operator past all occupied orbitals between positions *p* and *q*:

    .. math::

        \text{sign} = (-1)^{\\sum_{k=\\min(p,q)+1}^{\\max(p,q)-1} n_k}

    Examples
    --------
    >>> import numpy as np
    >>> config = np.array([1, 1, 0, 1])
    >>> numba_jw_sign_single(config, 0, 3)
    -1
    """
    lo = min(p, q) + 1
    hi = max(p, q)
    count = 0
    for k in range(lo, hi):
        count += config[k]
    return 1 - 2 * (count % 2)


@njit(cache=True)
def numba_jw_sign_double(
    config: np.ndarray, p: int, r: int, q: int, s: int
) -> int:
    """Compute the Jordan--Wigner sign for a double excitation a†_p a†_r a_s a_q.

    The operator ordering is right-to-left: first annihilate q, then s,
    then create r, then create p.  Each step accumulates a sign from the
    occupied orbitals it must anti-commute past (on the *current* state
    of the configuration after previous operations).

    Parameters
    ----------
    config : np.ndarray
        Occupation vector, shape ``(num_sites,)``, dtype int.
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

    Notes
    -----
    Each fermionic operator is translated into a Jordan--Wigner string
    that requires a Z-chain parity count.  The four operations are
    applied sequentially on a mutable copy of the configuration to
    capture the parity of occupied orbitals at each step.

    Examples
    --------
    >>> import numpy as np
    >>> config = np.array([1, 1, 0, 0])
    >>> numba_jw_sign_double(config, 2, 3, 0, 1)
    1
    """
    # Work on a copy so mutations do not affect caller
    state = config.copy()
    sign = 1

    # Step 1: annihilate q
    lo = min(0, q)
    count_q = 0
    for k in range(0, q):
        count_q += state[k]
    sign *= 1 - 2 * (count_q % 2)
    state[q] = 0

    # Step 2: annihilate s
    count_s = 0
    for k in range(0, s):
        count_s += state[k]
    sign *= 1 - 2 * (count_s % 2)
    state[s] = 0

    # Step 3: create r
    count_r = 0
    for k in range(0, r):
        count_r += state[k]
    sign *= 1 - 2 * (count_r % 2)
    state[r] = 1

    # Step 4: create p
    count_p = 0
    for k in range(0, p):
        count_p += state[k]
    sign *= 1 - 2 * (count_p % 2)
    state[p] = 1

    return sign
