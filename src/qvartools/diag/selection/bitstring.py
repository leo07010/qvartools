"""
Bitstring and basis-set utility functions for SKQD postprocessing.

Provides helpers for converting between bitstring and integer
representations, accumulating measurement results across Krylov steps,
filtering low-probability states, and computing basis-set overlap metrics.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

__all__ = [
    "bitstring_to_int",
    "int_to_bitstring",
    "get_basis_states_as_array",
    "calculate_cumulative_results",
    "filter_high_probability_states",
    "compute_basis_overlap",
    "estimate_ground_state_sparsity",
    "merge_basis_sets",
]


# ---------------------------------------------------------------------------
# Bitstring <-> integer conversions
# ---------------------------------------------------------------------------

def bitstring_to_int(bitstring: str) -> int:
    """Convert a binary bitstring to its integer representation.

    Parameters
    ----------
    bitstring : str
        String of ``'0'`` and ``'1'`` characters (e.g. ``"0110"``).

    Returns
    -------
    int
        Integer value of the bitstring.

    Examples
    --------
    >>> bitstring_to_int("0110")
    6
    """
    return int(bitstring, 2)


def int_to_bitstring(value: int, num_bits: int) -> str:
    """Convert an integer to a zero-padded binary bitstring.

    Parameters
    ----------
    value : int
        Non-negative integer to convert.
    num_bits : int
        Total width of the output string (zero-padded on the left).

    Returns
    -------
    str
        Binary string of length ``num_bits``.

    Examples
    --------
    >>> int_to_bitstring(6, 4)
    '0110'
    """
    return format(value, f"0{num_bits}b")


# ---------------------------------------------------------------------------
# Measurement-result helpers
# ---------------------------------------------------------------------------

def get_basis_states_as_array(
    measurement_results: Dict[str, int],
    num_qubits: int,  # noqa: ARG001 -- kept for API compatibility
) -> np.ndarray:
    """Convert measurement results to an array of unique basis-state integers.

    Parameters
    ----------
    measurement_results : dict of str to int
        Mapping from bitstring (e.g. ``"0110"``) to occurrence count.
    num_qubits : int
        Number of qubits.  Unused; kept for backward compatibility.

    Returns
    -------
    np.ndarray
        Sorted array of unique basis-state integers, dtype ``int64``.

    Examples
    --------
    >>> results = {"01": 5, "10": 3, "01": 2}
    >>> get_basis_states_as_array(results, num_qubits=2)
    array([1, 2])
    """
    states = [bitstring_to_int(bs) for bs in measurement_results]
    return np.array(sorted(set(states)), dtype=np.int64)


def calculate_cumulative_results(
    all_measurement_results: List[Dict[str, int]],
) -> List[Dict[str, int]]:
    """Calculate cumulative measurement results across Krylov steps.

    For step *k*, the cumulative results include all unique bitstrings
    from steps 0, 1, ..., *k* with their total counts.

    Parameters
    ----------
    all_measurement_results : list of dict
        One measurement dictionary per Krylov step, each mapping
        bitstring to count.

    Returns
    -------
    list of dict
        Cumulative measurement dictionaries.  Entry *k* contains the
        union of all bitstrings observed in steps 0 through *k*.

    Examples
    --------
    >>> step0 = {"00": 3, "01": 2}
    >>> step1 = {"01": 1, "10": 4}
    >>> cumulative = calculate_cumulative_results([step0, step1])
    >>> cumulative[1]
    {'00': 3, '01': 3, '10': 4}
    """
    cumulative: List[Dict[str, int]] = []
    all_counts: Dict[str, int] = {}

    for step_results in all_measurement_results:
        # Merge counts (immutable snapshot per step)
        for bitstring, count in step_results.items():
            all_counts[bitstring] = all_counts.get(bitstring, 0) + count

        # Store snapshot
        cumulative.append(dict(all_counts))

    return cumulative


def filter_high_probability_states(
    measurement_results: Dict[str, int],
    threshold: float = 0.0,
    max_states: int | None = None,
) -> Dict[str, int]:
    """Filter measurement results to keep only high-probability states.

    Parameters
    ----------
    measurement_results : dict of str to int
        Mapping from bitstring to occurrence count.
    threshold : float, optional
        Minimum empirical probability for a state to be retained
        (default ``0.0``, i.e. keep all).
    max_states : int or None, optional
        If not ``None``, keep at most this many states (the highest-count
        states are preferred).

    Returns
    -------
    dict of str to int
        Filtered measurement dictionary.

    Examples
    --------
    >>> counts = {"00": 90, "01": 5, "10": 3, "11": 2}
    >>> filter_high_probability_states(counts, threshold=0.05)
    {'00': 90, '01': 5}
    """
    total_counts = sum(measurement_results.values())
    if total_counts == 0:
        return {}

    # Compute probabilities
    probs = {
        bs: count / total_counts
        for bs, count in measurement_results.items()
    }

    # Filter by threshold
    filtered = {
        bs: count
        for bs, count in measurement_results.items()
        if probs[bs] >= threshold
    }

    # Limit number of states
    if max_states is not None and len(filtered) > max_states:
        sorted_states = sorted(
            filtered.items(), key=lambda x: x[1], reverse=True
        )
        filtered = dict(sorted_states[:max_states])

    return filtered


# ---------------------------------------------------------------------------
# Basis-set analysis
# ---------------------------------------------------------------------------

def compute_basis_overlap(
    basis1: np.ndarray,
    basis2: np.ndarray,
) -> float:
    """Compute the overlap fraction between two basis sets.

    Returns the fraction of states in ``basis1`` that are also present
    in ``basis2``.

    Parameters
    ----------
    basis1 : np.ndarray
        First basis as an array of state integers.
    basis2 : np.ndarray
        Second basis as an array of state integers.

    Returns
    -------
    float
        Overlap fraction in the range ``[0, 1]``.  Returns ``0.0``
        when ``basis1`` is empty.

    Examples
    --------
    >>> b1 = np.array([0, 1, 2, 3])
    >>> b2 = np.array([2, 3, 4, 5])
    >>> compute_basis_overlap(b1, b2)
    0.5
    """
    set1 = set(basis1.tolist())
    set2 = set(basis2.tolist())

    intersection = len(set1 & set2)
    return intersection / len(set1) if len(set1) > 0 else 0.0


def estimate_ground_state_sparsity(
    ground_state: np.ndarray,
    threshold: float = 1e-6,
) -> Dict[str, float]:
    """Estimate sparsity metrics of a ground-state wavefunction.

    Parameters
    ----------
    ground_state : np.ndarray
        Ground-state wavefunction vector, shape ``(dim,)``.
    threshold : float, optional
        Minimum probability ``|c_i|^2`` for a component to be counted
        as significant (default ``1e-6``).

    Returns
    -------
    dict
        Sparsity metrics with keys:

        - ``"n_significant"`` : int -- number of components above
          ``threshold``.
        - ``"sparsity_ratio"`` : float -- fraction of Hilbert space
          with significant weight.
        - ``"concentration"`` : float -- total probability weight in
          the top 10 %% of components.
        - ``"total_dimension"`` : int -- total Hilbert-space dimension.

    Examples
    --------
    >>> psi = np.array([0.9, 0.1, 0.0, 0.0])
    >>> stats = estimate_ground_state_sparsity(psi, threshold=1e-4)
    >>> stats["n_significant"]
    2
    """
    probs = np.abs(ground_state) ** 2
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum

    n_significant = int(np.sum(probs > threshold))
    sparsity_ratio = n_significant / len(ground_state)

    # Weight in top components
    sorted_probs = np.sort(probs)[::-1]
    n_top = max(1, len(sorted_probs) // 10)
    concentration = float(np.sum(sorted_probs[:n_top]))

    return {
        "n_significant": n_significant,
        "sparsity_ratio": float(sparsity_ratio),
        "concentration": concentration,
        "total_dimension": len(ground_state),
    }


def merge_basis_sets(*bases: np.ndarray) -> np.ndarray:
    """Merge multiple basis sets into one sorted, unique set.

    Parameters
    ----------
    *bases : np.ndarray
        Variable number of 1-D arrays of basis-state integers.

    Returns
    -------
    np.ndarray
        Sorted array of unique basis states, dtype ``int64``.

    Examples
    --------
    >>> b1 = np.array([0, 1, 2])
    >>> b2 = np.array([2, 3])
    >>> merge_basis_sets(b1, b2)
    array([0, 1, 2, 3])
    """
    all_states: set[int] = set()
    for basis in bases:
        all_states.update(basis.tolist())

    return np.array(sorted(all_states), dtype=np.int64)
