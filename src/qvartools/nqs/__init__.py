"""
nqs --- Neural quantum state architectures
===========================================

This subpackage provides neural quantum state (NQS) ansaetze for
variational Monte Carlo and related methods.  All architectures inherit
from :class:`NeuralQuantumState` and expose a uniform interface for
log-amplitudes, phases, and wavefunction evaluation.

Classes
-------
NeuralQuantumState
    Abstract base class for all NQS architectures.
DenseNQS
    Fully connected feedforward NQS (real or complex output).
SignedDenseNQS
    Dense NQS with explicit sign structure (amplitude + sign heads).
ComplexNQS
    Dense NQS with shared feature extractor for amplitude and phase.
RBMQuantumState
    Restricted Boltzmann Machine NQS (Carleo & Troyer, 2017).
AutoregressiveTransformer
    Autoregressive transformer NQS with alpha/beta spin channels.

Functions
---------
compile_nqs
    Apply ``torch.compile`` to an NQS model with graceful fallback.
"""

from qvartools.nqs.neural_state import NeuralQuantumState
from qvartools.nqs.architectures.complex_nqs import ComplexNQS
from qvartools.nqs.architectures.rbm import RBMQuantumState
from qvartools.nqs.architectures.dense import DenseNQS, SignedDenseNQS, compile_nqs
from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer

__all__ = [
    "NeuralQuantumState",
    "DenseNQS",
    "SignedDenseNQS",
    "ComplexNQS",
    "RBMQuantumState",
    "AutoregressiveTransformer",
    "compile_nqs",
]
