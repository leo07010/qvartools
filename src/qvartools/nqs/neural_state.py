"""
base --- Abstract neural quantum state base class
==================================================

Provides the :class:`NeuralQuantumState` ABC that every concrete NQS
architecture must implement.  Subclasses define :meth:`log_amplitude`
and :meth:`phase`; the base class assembles these into the full
wavefunction, probabilities, and normalised probabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "NeuralQuantumState",
]


class NeuralQuantumState(nn.Module, ABC):
    """Abstract base class for all neural quantum state ansaetze.

    Every subclass must implement :meth:`log_amplitude` and :meth:`phase`.
    The base class provides convenience methods for evaluating the full
    log-wavefunction, the wavefunction itself, Born-rule probabilities,
    and normalised probabilities over a discrete basis set.

    Parameters
    ----------
    num_sites : int
        Number of lattice / orbital sites (input dimensionality).
    local_dim : int, optional
        Dimension of the local Hilbert space on each site (default ``2``
        for spin-1/2 / qubit systems).
    complex_output : bool, optional
        If ``True`` the NQS represents a complex-valued wavefunction with
        a non-trivial phase network.  If ``False`` the phase is identically
        zero and :meth:`log_psi` returns a single real tensor.

    Attributes
    ----------
    num_sites : int
        Number of sites.
    local_dim : int
        Local Hilbert-space dimension.
    complex_output : bool
        Whether the NQS has a non-trivial phase.
    """

    def __init__(
        self,
        num_sites: int,
        local_dim: int = 2,
        complex_output: bool = False,
    ) -> None:
        super().__init__()
        if num_sites < 1:
            raise ValueError(f"num_sites must be >= 1, got {num_sites}")
        if local_dim < 2:
            raise ValueError(f"local_dim must be >= 2, got {local_dim}")
        self.num_sites: int = num_sites
        self.local_dim: int = local_dim
        self.complex_output: bool = complex_output

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log-amplitude ln|psi(x)| for each configuration.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Log-amplitudes, shape ``(batch,)``.
        """

    @abstractmethod
    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the phase arg(psi(x)) for each configuration.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Phases in radians, shape ``(batch,)``.  Must be identically
            zero when ``complex_output is False``.
        """

    # ------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------

    def log_psi(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the log-wavefunction.

        For real-valued NQS (``complex_output is False``), returns only the
        log-amplitude.  For complex-valued NQS, returns a tuple of
        (log_amplitude, phase).

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            If ``complex_output is False``: log-amplitude tensor of shape
            ``(batch,)``.
            If ``complex_output is True``: tuple ``(log_amp, phase)`` each
            of shape ``(batch,)``.
        """
        log_amp = self.log_amplitude(x)
        if not self.complex_output:
            return log_amp
        return log_amp, self.phase(x)

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the full wavefunction psi(x).

        Computes ``exp(log_amp) * exp(i * phase)``.  When
        ``complex_output is False`` the result is real-valued
        (dtype matches input); otherwise it is complex.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Wavefunction values, shape ``(batch,)``.  Complex dtype when
            ``complex_output is True``.
        """
        log_amp = self.log_amplitude(x)
        if not self.complex_output:
            return torch.exp(log_amp)
        phi = self.phase(x)
        amplitude = torch.exp(log_amp)
        return amplitude * torch.exp(1j * phi.to(torch.complex64))

    def probability(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Born-rule probability |psi(x)|^2 (unnormalised).

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Unnormalised probabilities ``exp(2 * log_amplitude(x))``,
            shape ``(batch,)``.
        """
        return torch.exp(2.0 * self.log_amplitude(x))

    def normalized_probability(
        self, x: torch.Tensor, basis_set: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalised Born-rule probabilities over a basis set.

        The normalisation constant Z is computed as the sum of
        ``|psi(s)|^2`` over every configuration *s* in *basis_set*.

        Parameters
        ----------
        x : torch.Tensor
            Configurations to evaluate, shape ``(batch, num_sites)``.
        basis_set : torch.Tensor
            Complete (or reference) set of configurations used to compute
            the partition function, shape ``(n_basis, num_sites)``.

        Returns
        -------
        torch.Tensor
            Normalised probabilities, shape ``(batch,)``.  Each entry is
            ``|psi(x_i)|^2 / Z``.
        """
        log_amp_x = self.log_amplitude(x)
        log_amp_basis = self.log_amplitude(basis_set)

        # Compute log(Z) = log(sum exp(2 * log_amp)) via logsumexp for
        # numerical stability.
        log_z = torch.logsumexp(2.0 * log_amp_basis, dim=0)
        return torch.exp(2.0 * log_amp_x - log_z)

    @staticmethod
    def encode_configuration(config: torch.Tensor) -> torch.Tensor:
        """Convert a configuration tensor to float for network input.

        Parameters
        ----------
        config : torch.Tensor
            Configuration tensor of any integer or float dtype, shape
            ``(..., num_sites)``.

        Returns
        -------
        torch.Tensor
            Float tensor with the same shape, dtype ``torch.float32``.
        """
        return config.to(torch.float32)

    # ------------------------------------------------------------------
    # Forward (default delegates to log_psi)
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass --- delegates to :meth:`log_psi`.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Same as :meth:`log_psi`.
        """
        return self.log_psi(x)
