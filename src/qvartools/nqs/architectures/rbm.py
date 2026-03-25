"""
rbm --- Restricted Boltzmann Machine neural quantum state
=========================================================

Provides :class:`RBMQuantumState`, an RBM-based NQS following the approach
of Carleo & Troyer (Science, 2017).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from qvartools.nqs.neural_state import NeuralQuantumState

__all__ = [
    "RBMQuantumState",
]


# ---------------------------------------------------------------------------
# RBMQuantumState
# ---------------------------------------------------------------------------


class RBMQuantumState(NeuralQuantumState):
    r"""Restricted Boltzmann Machine neural quantum state.

    Implements the RBM ansatz of Carleo & Troyer (Science, 2017):

    .. math::

        \psi(\mathbf{x}) = \exp\!\Bigl(\sum_j a_j x_j\Bigr)
        \prod_i \cosh\!\Bigl(b_i + \sum_j W_{ij} x_j\Bigr)

    When ``complex_weights`` is ``True``, the parameters ``a``, ``b``, and
    ``W`` are complex-valued, and the wavefunction acquires a non-trivial
    phase.

    Parameters
    ----------
    num_sites : int
        Number of visible units (lattice / orbital sites).
    num_hidden : int, optional
        Number of hidden units (default ``num_sites``).
    complex_weights : bool, optional
        If ``True``, use complex-valued RBM parameters to represent a
        complex wavefunction (default ``False``).

    Attributes
    ----------
    a_real : nn.Parameter
        Real part of the visible bias, shape ``(num_sites,)``.
    a_imag : nn.Parameter or None
        Imaginary part of the visible bias (only if ``complex_weights``).
    b_real : nn.Parameter
        Real part of the hidden bias, shape ``(num_hidden,)``.
    b_imag : nn.Parameter or None
        Imaginary part of the hidden bias (only if ``complex_weights``).
    W_real : nn.Parameter
        Real part of the weight matrix, shape ``(num_hidden, num_sites)``.
    W_imag : nn.Parameter or None
        Imaginary part of the weight matrix (only if ``complex_weights``).

    Examples
    --------
    >>> rbm = RBMQuantumState(num_sites=10, num_hidden=20)
    >>> x = torch.randint(0, 2, (8, 10)).float()
    >>> log_amp = rbm.log_amplitude(x)  # shape (8,)
    """

    def __init__(
        self,
        num_sites: int,
        num_hidden: Optional[int] = None,
        complex_weights: bool = False,
    ) -> None:
        super().__init__(
            num_sites=num_sites,
            local_dim=2,
            complex_output=complex_weights,
        )
        if num_hidden is None:
            num_hidden = num_sites
        self.num_hidden: int = num_hidden
        self.complex_weights: bool = complex_weights

        # Visible bias
        self.a_real: nn.Parameter = nn.Parameter(
            torch.randn(num_sites) * 0.01
        )
        # Hidden bias
        self.b_real: nn.Parameter = nn.Parameter(
            torch.randn(num_hidden) * 0.01
        )
        # Weight matrix
        self.W_real: nn.Parameter = nn.Parameter(
            torch.randn(num_hidden, num_sites) * (1.0 / math.sqrt(num_sites))
        )

        # Complex parts (optional)
        self.a_imag: Optional[nn.Parameter] = None
        self.b_imag: Optional[nn.Parameter] = None
        self.W_imag: Optional[nn.Parameter] = None

        if complex_weights:
            self.a_imag = nn.Parameter(torch.randn(num_sites) * 0.01)
            self.b_imag = nn.Parameter(torch.randn(num_hidden) * 0.01)
            self.W_imag = nn.Parameter(
                torch.randn(num_hidden, num_sites)
                * (1.0 / math.sqrt(num_sites))
            )

    def _theta(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the pre-activation :math:`\theta_i = b_i + \sum_j W_{ij} x_j`.

        When ``complex_weights`` is ``True``, returns a complex tensor.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Pre-activations, shape ``(batch, num_hidden)``.  Complex dtype
            when ``complex_weights`` is ``True``.
        """
        x = self.encode_configuration(x)
        # theta_real = b_real + x @ W_real^T,  shape (batch, num_hidden)
        theta_real = torch.addmm(self.b_real, x, self.W_real.t())

        if not self.complex_weights:
            return theta_real

        # theta_imag = b_imag + x @ W_imag^T
        assert self.b_imag is not None
        assert self.W_imag is not None
        theta_imag = torch.addmm(self.b_imag, x, self.W_imag.t())
        return torch.complex(theta_real, theta_imag)

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the log-amplitude of the RBM wavefunction.

        For real weights:

        .. math::

            \ln|\psi(\mathbf{x})| = \mathrm{Re}(\mathbf{a}) \cdot \mathbf{x}
            + \sum_i \ln\cosh(\theta_i)

        For complex weights, takes the real part of the full
        log-wavefunction.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Log-amplitudes, shape ``(batch,)``.
        """
        x_enc = self.encode_configuration(x)
        theta = self._theta(x)

        if not self.complex_weights:
            # Real case: a · x + sum log(cosh(theta))
            visible_term = torch.mv(x_enc, self.a_real)  # (batch,)
            # log(cosh(z)) = |z| + log(1 + exp(-2|z|)) - log(2)
            # for numerical stability
            log_cosh = _log_cosh_real(theta)  # (batch, num_hidden)
            return visible_term + log_cosh.sum(dim=-1)

        # Complex case: take real part of full log-psi
        assert self.a_imag is not None
        a_complex = torch.complex(self.a_real, self.a_imag)
        visible_term = torch.mv(x_enc, a_complex.real)  # Re(a) . x

        # For complex theta, log|cosh(theta)| = Re(log(cosh(theta)))
        log_cosh = _log_cosh_complex(theta)  # (batch, num_hidden)
        return visible_term + log_cosh.sum(dim=-1)

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the wavefunction phase.

        For real weights the phase is identically zero.  For complex
        weights, the phase is the imaginary part of the full
        log-wavefunction:

        .. math::

            \arg\psi(\mathbf{x}) = \mathrm{Im}(\mathbf{a}) \cdot \mathbf{x}
            + \sum_i \mathrm{Im}\bigl(\ln\cosh(\theta_i)\bigr)

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Phases in radians, shape ``(batch,)``.
        """
        if not self.complex_weights:
            return torch.zeros(
                x.shape[0], device=x.device, dtype=torch.float32
            )

        x_enc = self.encode_configuration(x)
        theta = self._theta(x)

        assert self.a_imag is not None
        # Im(a) . x
        visible_phase = torch.mv(x_enc, self.a_imag)  # (batch,)

        # Im(log(cosh(theta)))
        # cosh(a+bi) = cosh(a)cos(b) + i sinh(a)sin(b)
        # arg(cosh(a+bi)) = atan2(sinh(a)sin(b), cosh(a)cos(b))
        theta_real = theta.real
        theta_imag = theta.imag
        hidden_phase = torch.atan2(
            torch.sinh(theta_real) * torch.sin(theta_imag),
            torch.cosh(theta_real) * torch.cos(theta_imag),
        )  # (batch, num_hidden)

        return visible_phase + hidden_phase.sum(dim=-1)


# ---------------------------------------------------------------------------
# Numerically stable log-cosh helpers
# ---------------------------------------------------------------------------


def _log_cosh_real(x: torch.Tensor) -> torch.Tensor:
    """Compute log(cosh(x)) in a numerically stable way for real x.

    Parameters
    ----------
    x : torch.Tensor
        Real-valued input tensor.

    Returns
    -------
    torch.Tensor
        ``log(cosh(x))``, same shape as input.
    """
    # log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
    abs_x = torch.abs(x)
    return abs_x + torch.nn.functional.softplus(-2.0 * abs_x) - math.log(2.0)


def _log_cosh_complex(z: torch.Tensor) -> torch.Tensor:
    """Compute Re(log(cosh(z))) for complex z (log-amplitude contribution).

    Parameters
    ----------
    z : torch.Tensor
        Complex-valued input tensor.

    Returns
    -------
    torch.Tensor
        Real part of ``log(cosh(z))``, same shape as input, real dtype.
    """
    # |cosh(a+bi)|^2 = cosh^2(a) cos^2(b) + sinh^2(a) sin^2(b)
    #                 = (cosh(2a) + cos(2b)) / 2
    # log|cosh(z)| = 0.5 * log((cosh(2a) + cos(2b)) / 2)
    a = z.real
    b = z.imag
    # Use stable version: cosh(2a) ~ exp(2|a|)/2 for large |a|
    log_cosh_2a = _log_cosh_real(2.0 * a)  # log(cosh(2a))
    # log|cosh(z)| = 0.5 * (log_cosh_2a + log(1 + cos(2b)/cosh(2a)))
    # Simplify: 0.5 * log((cosh(2a) + cos(2b))/2)
    #         = 0.5 * (log(cosh(2a) + cos(2b)) - log(2))
    inner = torch.exp(log_cosh_2a) + torch.cos(2.0 * b)
    # Clamp for safety (should always be positive but floating point)
    inner = torch.clamp(inner, min=1e-30)
    return 0.5 * (torch.log(inner) - math.log(2.0))
