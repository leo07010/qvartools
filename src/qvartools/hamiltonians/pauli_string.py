"""
pauli_string — Lightweight Pauli string representation
=======================================================

Provides ``PauliString``, a single tensor-product Pauli operator
(I, X, Y, Z per qubit) with a complex coefficient.  The class supports
application to computational-basis states and diagonal detection.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

__all__ = [
    "PauliString",
]


class PauliString:
    """A single Pauli string (tensor product of I, X, Y, Z) with coefficient.

    Parameters
    ----------
    paulis : list of str
        One-character Pauli labels for each qubit.  Allowed values are
        ``"I"``, ``"X"``, ``"Y"``, ``"Z"``.
    coefficient : complex, optional
        Multiplicative coefficient (default ``1.0``).

    Attributes
    ----------
    paulis : list of str
        The Pauli labels.
    coefficient : complex
        The coefficient.
    num_qubits : int
        Number of qubits the string acts on.

    Raises
    ------
    ValueError
        If any label in *paulis* is not in ``{"I", "X", "Y", "Z"}``.

    Examples
    --------
    >>> ps = PauliString(["X", "Z", "I"], coefficient=0.5)
    >>> new_config, phase = ps.apply(torch.tensor([0, 1, 0]))
    """

    _VALID_PAULIS = frozenset({"I", "X", "Y", "Z"})

    def __init__(
        self, paulis: List[str], coefficient: complex = 1.0
    ) -> None:
        for p in paulis:
            if p not in self._VALID_PAULIS:
                raise ValueError(
                    f"Invalid Pauli label '{p}'. Must be one of {sorted(self._VALID_PAULIS)}."
                )
        self.paulis: List[str] = list(paulis)
        self.coefficient: complex = complex(coefficient)
        self.num_qubits: int = len(paulis)

    def apply(
        self, config: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], complex]:
        """Apply this Pauli string to a computational-basis state.

        The computational basis is ``|0⟩, |1⟩`` per qubit.  The rules are:

        * **I** — identity, no change.
        * **X** — bit flip: ``|0⟩ → |1⟩``, ``|1⟩ → |0⟩``.
        * **Y** — bit flip with phase: ``|0⟩ → i|1⟩``, ``|1⟩ → −i|0⟩``.
        * **Z** — phase flip: ``|0⟩ → |0⟩``, ``|1⟩ → −|1⟩``.

        Parameters
        ----------
        config : torch.Tensor
            Input configuration, shape ``(num_qubits,)`` with entries
            in ``{0, 1}``.

        Returns
        -------
        new_config : torch.Tensor or None
            The resulting configuration after applying the string.
            ``None`` is never returned for valid inputs (kept in the
            signature for forward-compatibility with annihilation
            operators).
        phase : complex
            The accumulated phase (including ``self.coefficient``).

        Examples
        --------
        >>> ps = PauliString(["X", "Z"], coefficient=1.0)
        >>> new_cfg, phase = ps.apply(torch.tensor([0, 1]))
        >>> new_cfg
        tensor([1, 1])
        >>> phase
        (-1+0j)
        """
        new_config = config.clone()
        phase: complex = self.coefficient

        for qubit, pauli in enumerate(self.paulis):
            bit = int(config[qubit].item())
            if pauli == "I":
                continue
            elif pauli == "X":
                new_config[qubit] = 1 - bit
            elif pauli == "Y":
                new_config[qubit] = 1 - bit
                # Y|0⟩ = i|1⟩,  Y|1⟩ = −i|0⟩
                phase *= 1j if bit == 0 else -1j
            elif pauli == "Z":
                if bit == 1:
                    phase *= -1

        return new_config, phase

    def is_diagonal(self) -> bool:
        """Return ``True`` if the Pauli string is diagonal (only I and Z).

        Returns
        -------
        bool

        Examples
        --------
        >>> PauliString(["Z", "I", "Z"]).is_diagonal()
        True
        >>> PauliString(["X", "I"]).is_diagonal()
        False
        """
        return all(p in ("I", "Z") for p in self.paulis)

    def __repr__(self) -> str:
        pauli_str = "".join(self.paulis)
        return f"PauliString({pauli_str!r}, coeff={self.coefficient})"
