"""
pauli_mapping — Jordan--Wigner transformation to Pauli representation
======================================================================

Converts second-quantised fermionic Hamiltonians to a qubit (Pauli)
representation for use with quantum circuits (CUDA-Q ``exp_pauli``).

Also provides direct Pauli construction for spin Hamiltonians
(Heisenberg model) matching the NVIDIA CUDA-Q SKQD tutorial format.

References
----------
.. [1] Jordan & Wigner, *Z. Phys.* **47**, 631 (1928).
.. [2] NVIDIA CUDA-Q SKQD tutorial:
   https://nvidia.github.io/cuda-quantum/latest/applications/python/skqd.html
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "PauliSum",
    "one_body_op",
    "two_body_op",
    "molecular_hamiltonian_to_pauli",
    "heisenberg_hamiltonian_pauli",
]


# ---------------------------------------------------------------------------
# Pauli algebra
# ---------------------------------------------------------------------------

# Single-qubit Pauli multiplication table: (row, col) -> (phase, result)
# Phase is a power of i: 0=1, 1=i, 2=-1, 3=-i
_PAULI_MULT = {
    ("I", "I"): (0, "I"), ("I", "X"): (0, "X"), ("I", "Y"): (0, "Y"), ("I", "Z"): (0, "Z"),
    ("X", "I"): (0, "X"), ("X", "X"): (0, "I"), ("X", "Y"): (1, "Z"), ("X", "Z"): (3, "Y"),
    ("Y", "I"): (0, "Y"), ("Y", "X"): (3, "Z"), ("Y", "Y"): (0, "I"), ("Y", "Z"): (1, "X"),
    ("Z", "I"): (0, "Z"), ("Z", "X"): (1, "Y"), ("Z", "Y"): (3, "X"), ("Z", "Z"): (0, "I"),
}

_PHASE_TO_COMPLEX = {0: 1.0, 1: 1j, 2: -1.0, 3: -1j}


def _multiply_pauli_strings(s1: str, s2: str) -> Tuple[complex, str]:
    """Multiply two multi-qubit Pauli strings site-by-site.

    Applies the single-qubit Pauli multiplication table independently
    on each qubit position and accumulates the overall phase factor.

    Parameters
    ----------
    s1 : str
        First Pauli string, e.g. ``"XIZI"``.
    s2 : str
        Second Pauli string, same length as *s1*.

    Returns
    -------
    phase : complex
        Accumulated phase from the per-site multiplication (a power of *i*).
    result_string : str
        Resulting Pauli string.

    Raises
    ------
    AssertionError
        If ``len(s1) != len(s2)``.
    """
    assert len(s1) == len(s2), f"Pauli string length mismatch: {len(s1)} vs {len(s2)}"
    total_phase = 0  # accumulated power of i
    result = []
    for p1, p2 in zip(s1, s2):
        phase_power, res = _PAULI_MULT[(p1, p2)]
        total_phase = (total_phase + phase_power) % 4
        result.append(res)
    return _PHASE_TO_COMPLEX[total_phase], "".join(result)


class PauliSum:
    """Sparse representation of a sum of weighted Pauli strings.

    Stores ``{pauli_string: complex_coefficient}`` and supports addition,
    scalar multiplication, and operator multiplication (Pauli product).

    Parameters
    ----------
    n_qubits : int
        Number of qubits the operator acts on.

    Attributes
    ----------
    terms : dict of str to complex
        Mapping from Pauli string labels to their coefficients.
    n_qubits : int
        Number of qubits.

    Examples
    --------
    >>> ps = PauliSum(2)
    >>> ps.add_term(0.5, "ZI")
    >>> ps.add_term(-0.3, "IX")
    >>> ps.terms
    {'ZI': 0.5, 'IX': -0.3}
    """

    __slots__ = ("terms", "n_qubits")

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.terms: Dict[str, complex] = {}

    def add_term(self, coeff: complex, pauli_string: str) -> None:
        """Add ``coeff * pauli_string`` to this operator.

        If the Pauli string already exists, the coefficient is accumulated.
        Terms with ``|coeff| < 1e-18`` are silently dropped.

        Parameters
        ----------
        coeff : complex
            Coefficient of the Pauli term.
        pauli_string : str
            Pauli string label, e.g. ``"XZIY"``.  Must have length
            equal to ``n_qubits``.
        """
        if abs(coeff) < 1e-18:
            return
        assert len(pauli_string) == self.n_qubits
        self.terms[pauli_string] = self.terms.get(pauli_string, 0.0) + coeff

    def __iadd__(self, other: "PauliSum") -> "PauliSum":
        """Add all terms from *other* into this ``PauliSum`` in-place.

        Parameters
        ----------
        other : PauliSum
            Operator whose terms are added.

        Returns
        -------
        PauliSum
            ``self``, modified in-place.
        """
        for ps, c in other.terms.items():
            self.terms[ps] = self.terms.get(ps, 0.0) + c
        return self

    def scale(self, scalar: complex) -> "PauliSum":
        """Return a new ``PauliSum`` with all coefficients multiplied by *scalar*.

        Parameters
        ----------
        scalar : complex
            Multiplicative factor.

        Returns
        -------
        PauliSum
            New operator with scaled coefficients.
        """
        result = PauliSum(self.n_qubits)
        for ps, c in self.terms.items():
            result.terms[ps] = c * scalar
        return result

    def multiply(self, other: "PauliSum") -> "PauliSum":
        """Compute the operator product ``self * other``.

        Each pair of Pauli strings is multiplied site-by-site using the
        Pauli multiplication table, accumulating phases.

        Parameters
        ----------
        other : PauliSum
            Right-hand-side operator.

        Returns
        -------
        PauliSum
            Product operator.
        """
        result = PauliSum(self.n_qubits)
        for ps1, c1 in self.terms.items():
            for ps2, c2 in other.terms.items():
                phase, ps_result = _multiply_pauli_strings(ps1, ps2)
                coeff = c1 * c2 * phase
                if abs(coeff) > 1e-18:
                    result.terms[ps_result] = result.terms.get(ps_result, 0.0) + coeff
        return result

    def simplify(self, threshold: float = 1e-15) -> None:
        """Remove terms whose absolute coefficient is below *threshold*.

        Parameters
        ----------
        threshold : float, optional
            Magnitude threshold (default ``1e-15``).
        """
        self.terms = {ps: c for ps, c in self.terms.items() if abs(c) > threshold}

    def to_real_lists(self, threshold: float = 1e-12) -> Tuple[List[float], List[str], float]:
        """Export as real-valued coefficient lists, separating the constant.

        The all-identity Pauli string is extracted as a scalar constant.
        Imaginary parts are stripped (they arise from floating-point noise
        in the Jordan--Wigner algebra).

        Parameters
        ----------
        threshold : float, optional
            Magnitude threshold below which terms are dropped
            (default ``1e-12``).

        Returns
        -------
        coefficients : list of float
            Real coefficients (excluding the identity term).
        pauli_words : list of str
            Corresponding Pauli string labels.
        constant : float
            Coefficient of the all-identity string (includes nuclear
            repulsion when used with molecular Hamiltonians).

        Warns
        -----
        UserWarning
            If any term has ``|imag| / |real| > 1e-6``.
        """
        self.simplify(threshold)
        identity = "I" * self.n_qubits

        coefficients: List[float] = []
        pauli_words: List[str] = []
        constant = 0.0

        for ps, c in self.terms.items():
            # Warn if imaginary part is suspiciously large relative to real part
            if abs(c.real) > 1e-15 and abs(c.imag) / abs(c.real) > 1e-6:
                warnings.warn(
                    f"Pauli term '{ps}' has imaginary/real ratio "
                    f"{abs(c.imag) / abs(c.real):.2e} (coeff={c}). "
                    "Stripping imaginary part -- verify integrals are real.",
                    stacklevel=2,
                )
            real_c = c.real
            if abs(real_c) < threshold:
                continue
            if ps == identity:
                constant = real_c
            else:
                coefficients.append(real_c)
                pauli_words.append(ps)

        return coefficients, pauli_words, constant


# ---------------------------------------------------------------------------
# Jordan-Wigner elementary operators
# ---------------------------------------------------------------------------

def _identity_string(n_qubits: int) -> str:
    """Return the all-identity Pauli string of length *n_qubits*.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    str
        A string of ``"I"`` characters, e.g. ``"IIII"``.
    """
    return "I" * n_qubits


def _single_pauli(n_qubits: int, qubit: int, pauli: str) -> str:
    """Create a Pauli string with *pauli* on *qubit* and ``I`` elsewhere.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits.
    qubit : int
        Index of the target qubit.
    pauli : str
        Single-character Pauli label (``"X"``, ``"Y"``, or ``"Z"``).

    Returns
    -------
    str
        Pauli string of length *n_qubits*.
    """
    chars = ["I"] * n_qubits
    chars[qubit] = pauli
    return "".join(chars)


def _z_chain_string(n_qubits: int, start: int, end: int) -> str:
    """Create a Pauli string with ``Z`` on qubits ``[start, end)`` and ``I`` elsewhere.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits.
    start : int
        First qubit index with ``Z``.
    end : int
        One past the last qubit index with ``Z``.

    Returns
    -------
    str
        Pauli string of length *n_qubits*.
    """
    chars = ["I"] * n_qubits
    for q in range(start, end):
        chars[q] = "Z"
    return "".join(chars)


def one_body_op(p: int, q: int, n_qubits: int) -> PauliSum:
    r"""Jordan--Wigner transformation of the one-body operator :math:`a^\dagger_p a_q`.

    Parameters
    ----------
    p : int
        Creation orbital index.
    q : int
        Annihilation orbital index.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    PauliSum
        Qubit representation of :math:`a^\dagger_p a_q`.

    Notes
    -----
    * ``p == q``: number operator :math:`(I - Z_p) / 2`.
    * ``p < q``: :math:`\tfrac{1}{4}(XX + YY + iXY - iYX) \prod_{k=p+1}^{q-1} Z_k`.
    * ``p > q``: Hermitian conjugate of the ``(q, p)`` case.

    Examples
    --------
    >>> op = one_body_op(0, 0, 2)
    >>> sorted(op.terms.items())
    [('II', 0.5), ('ZI', -0.5)]
    """
    result = PauliSum(n_qubits)

    if p == q:
        # Number operator: (I - Z_p) / 2
        result.add_term(0.5, _identity_string(n_qubits))
        result.add_term(-0.5, _single_pauli(n_qubits, p, "Z"))
        return result

    # Ensure p < q, handle conjugate
    if p > q:
        conj = one_body_op(q, p, n_qubits)
        # Hermitian conjugate: (a+_q a_p)+ = a+_p a_q
        # For real Hamiltonians, h_pq = h_qp, so we just return the conjugate
        result_conj = PauliSum(n_qubits)
        for ps, c in conj.terms.items():
            result_conj.add_term(c.conjugate(), ps)
        return result_conj

    # p < q case
    # Build base strings with Z chain between p+1 and q-1
    base = ["I"] * n_qubits
    for k in range(p + 1, q):
        base[k] = "Z"

    # XX term
    xx = list(base)
    xx[p] = "X"
    xx[q] = "X"
    result.add_term(0.25, "".join(xx))

    # YY term
    yy = list(base)
    yy[p] = "Y"
    yy[q] = "Y"
    result.add_term(0.25, "".join(yy))

    # XY term (coefficient +i/4)
    xy = list(base)
    xy[p] = "X"
    xy[q] = "Y"
    result.add_term(0.25j, "".join(xy))

    # YX term (coefficient -i/4)
    yx = list(base)
    yx[p] = "Y"
    yx[q] = "X"
    result.add_term(-0.25j, "".join(yx))

    return result


def two_body_op(p: int, q: int, r: int, s: int, n_qubits: int) -> PauliSum:
    r"""Jordan--Wigner transformation of :math:`a^\dagger_p a^\dagger_r a_s a_q`.

    Computed as a product of one-body operators with a sign correction:

    .. math::

        a^\dagger_p a^\dagger_r a_s a_q
        = (a^\dagger_p a_q)(a^\dagger_r a_s)
          - \delta_{qr}\, a^\dagger_p a_s

    Parameters
    ----------
    p : int
        First creation orbital index.
    q : int
        First annihilation orbital index.
    r : int
        Second creation orbital index.
    s : int
        Second annihilation orbital index.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    PauliSum
        Qubit representation of the two-body operator.
    """
    # Compute (a+_p a_q) * (a+_r a_s)
    op_pq = one_body_op(p, q, n_qubits)
    op_rs = one_body_op(r, s, n_qubits)
    result = op_pq.multiply(op_rs)

    # Subtract delta_{qr} * (a+_p a_s) correction
    if q == r:
        op_ps = one_body_op(p, s, n_qubits)
        for ps_str, c in op_ps.terms.items():
            result.terms[ps_str] = result.terms.get(ps_str, 0.0) - c

    result.simplify()
    return result


# ---------------------------------------------------------------------------
# Full Hamiltonian transformations
# ---------------------------------------------------------------------------

def molecular_hamiltonian_to_pauli(
    h1e: np.ndarray,
    h2e: np.ndarray,
    nuclear_repulsion: float,
    n_orbitals: int,
) -> Tuple[List[float], List[str], float]:
    r"""Convert molecular integrals to Pauli representation via Jordan--Wigner.

    Qubit ordering matches the ``MolecularHamiltonian`` convention:
    qubits ``0..n_orb-1`` are alpha spin-orbitals and qubits
    ``n_orb..2*n_orb-1`` are beta spin-orbitals.

    Parameters
    ----------
    h1e : np.ndarray
        One-electron integrals in the MO basis, shape
        ``(n_orbitals, n_orbitals)``.
    h2e : np.ndarray
        Two-electron integrals in chemist's notation ``(pq|rs)``,
        shape ``(n_orbitals, n_orbitals, n_orbitals, n_orbitals)``.
    nuclear_repulsion : float
        Nuclear repulsion energy in Hartree.
    n_orbitals : int
        Number of spatial orbitals.

    Returns
    -------
    coefficients : list of float
        Real Pauli-term coefficients (excluding the identity term).
    pauli_words : list of str
        Corresponding Pauli string labels, each of length ``2 * n_orbitals``.
    constant_energy : float
        Sum of the identity Pauli coefficient and nuclear repulsion.

    Notes
    -----
    The full second-quantised Hamiltonian is

    .. math::

        H = E_\text{nuc}
          + \sum_{pq\sigma} h_{pq}\, a^\dagger_{p\sigma} a_{q\sigma}
          + \tfrac{1}{2} \sum_{pqrs\sigma\tau}
              (pq|rs)\, a^\dagger_{p\sigma} a^\dagger_{r\tau}
              a_{s\tau} a_{q\sigma}

    Each second-quantised operator is mapped to Pauli strings via
    :func:`one_body_op` and :func:`two_body_op`.

    Examples
    --------
    >>> import numpy as np
    >>> h1 = np.array([[-1.0, 0.0], [0.0, -0.5]])
    >>> h2 = np.zeros((2, 2, 2, 2))
    >>> coeffs, words, const = molecular_hamiltonian_to_pauli(h1, h2, 0.7, 2)
    """
    n_qubits = 2 * n_orbitals
    h_pauli = PauliSum(n_qubits)

    # Nuclear repulsion as identity term
    h_pauli.add_term(nuclear_repulsion, _identity_string(n_qubits))

    # --- One-body terms ---
    # H_1 = sum_{pq,sigma} h_pq a+_{p,sigma} a_{q,sigma}
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            if abs(h1e[p, q]) < 1e-15:
                continue
            # Alpha spin: qubit indices p, q
            op_alpha = one_body_op(p, q, n_qubits)
            h_pauli += op_alpha.scale(h1e[p, q])

            # Beta spin: qubit indices p + n_orbitals, q + n_orbitals
            op_beta = one_body_op(p + n_orbitals, q + n_orbitals, n_qubits)
            h_pauli += op_beta.scale(h1e[p, q])

    # --- Two-body terms ---
    # H_2 = 1/2 sum_{pqrs,sigma,tau} (pq|rs) a+_{p,sigma} a+_{r,tau} a_{s,tau} a_{q,sigma}
    # Chemist notation: (pq|rs) = <pr|qs> (physicist)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    coeff = 0.5 * h2e[p, q, r, s]
                    if abs(coeff) < 1e-15:
                        continue

                    # alpha-alpha: a+_{p,a} a+_{r,a} a_{s,a} a_{q,a}
                    op_aa = two_body_op(p, q, r, s, n_qubits)
                    h_pauli += op_aa.scale(coeff)

                    # beta-beta: a+_{p,b} a+_{r,b} a_{s,b} a_{q,b}
                    pb = p + n_orbitals
                    qb = q + n_orbitals
                    rb = r + n_orbitals
                    sb = s + n_orbitals
                    op_bb = two_body_op(pb, qb, rb, sb, n_qubits)
                    h_pauli += op_bb.scale(coeff)

                    # alpha-beta: a+_{p,a} a+_{r,b} a_{s,b} a_{q,a}
                    op_ab = two_body_op(p, q, r + n_orbitals, s + n_orbitals, n_qubits)
                    h_pauli += op_ab.scale(coeff)

                    # beta-alpha: a+_{p,b} a+_{r,a} a_{s,a} a_{q,b}
                    op_ba = two_body_op(p + n_orbitals, q + n_orbitals, r, s, n_qubits)
                    h_pauli += op_ba.scale(coeff)

    h_pauli.simplify()

    coefficients, pauli_words, constant = h_pauli.to_real_lists()

    # constant includes nuclear repulsion + identity Pauli terms
    return coefficients, pauli_words, constant


def heisenberg_hamiltonian_pauli(
    n_spins: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    hx: np.ndarray | None = None,
    hy: np.ndarray | None = None,
    hz: np.ndarray | None = None,
) -> Tuple[List[float], List[str], float]:
    r"""Heisenberg spin-chain Hamiltonian in Pauli string representation.

    Constructs the Pauli decomposition of

    .. math::

        H = \sum_i \bigl[J_x X_i X_{i+1} + J_y Y_i Y_{i+1}
                         + J_z Z_i Z_{i+1}\bigr]
          + \sum_i \bigl[h^x_i X_i + h^y_i Y_i + h^z_i Z_i\bigr]

    The output format matches the NVIDIA CUDA-Q SKQD tutorial.

    Parameters
    ----------
    n_spins : int
        Number of spins (qubits).
    Jx : float, optional
        XX coupling constant (default ``1.0``).
    Jy : float, optional
        YY coupling constant (default ``1.0``).
    Jz : float, optional
        ZZ coupling constant (default ``1.0``).
    hx : np.ndarray or None, optional
        External field in x-direction, length ``n_spins``
        (default ``np.ones(n_spins)``).
    hy : np.ndarray or None, optional
        External field in y-direction, length ``n_spins``
        (default ``np.ones(n_spins)``).
    hz : np.ndarray or None, optional
        External field in z-direction, length ``n_spins``
        (default ``np.ones(n_spins)``).

    Returns
    -------
    coefficients : list of float
        Real Pauli-term coefficients.
    pauli_words : list of str
        Corresponding Pauli string labels, each of length ``n_spins``.
    constant : float
        Constant energy offset (always ``0.0`` for this model).

    Examples
    --------
    >>> coeffs, words, const = heisenberg_hamiltonian_pauli(3, Jx=1.0, Jy=1.0, Jz=1.0)
    >>> len(words) > 0
    True
    """
    if hx is None:
        hx = np.ones(n_spins)
    if hy is None:
        hy = np.ones(n_spins)
    if hz is None:
        hz = np.ones(n_spins)

    coefficients: List[float] = []
    pauli_words: List[str] = []

    # Nearest-neighbor interactions
    for i in range(n_spins - 1):
        j = i + 1
        if abs(Jx) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "X"
            ps[j] = "X"
            coefficients.append(Jx)
            pauli_words.append("".join(ps))
        if abs(Jy) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "Y"
            ps[j] = "Y"
            coefficients.append(Jy)
            pauli_words.append("".join(ps))
        if abs(Jz) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "Z"
            ps[j] = "Z"
            coefficients.append(Jz)
            pauli_words.append("".join(ps))

    # External fields
    for i in range(n_spins):
        if abs(hx[i]) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "X"
            coefficients.append(float(hx[i]))
            pauli_words.append("".join(ps))
        if abs(hy[i]) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "Y"
            coefficients.append(float(hy[i]))
            pauli_words.append("".join(ps))
        if abs(hz[i]) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "Z"
            coefficients.append(float(hz[i]))
            pauli_words.append("".join(ps))

    return coefficients, pauli_words, 0.0
