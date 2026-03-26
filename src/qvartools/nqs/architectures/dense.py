"""
dense --- Dense feedforward NQS architectures
==============================================

Provides :class:`DenseNQS` (a standard fully connected NQS) and
:class:`SignedDenseNQS` (a dense NQS with explicit sign structure via
separate amplitude and sign heads sharing a feature extractor).

Also provides :func:`compile_nqs`, a utility that applies
``torch.compile`` to any NQS model with graceful fallback on error.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from qvartools.nqs.neural_state import NeuralQuantumState

__all__ = [
    "DenseNQS",
    "SignedDenseNQS",
    "compile_nqs",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# compile_nqs utility
# ---------------------------------------------------------------------------


def compile_nqs(
    model: nn.Module,
    mode: str = "reduce-overhead",
) -> nn.Module:
    """Apply ``torch.compile`` to an NQS model with graceful fallback.

    Parameters
    ----------
    model : nn.Module
        The neural quantum state model to compile.
    mode : str, optional
        Compilation mode passed to ``torch.compile``.  Common choices are
        ``"reduce-overhead"`` (default) and ``"max-autotune"``.

    Returns
    -------
    nn.Module
        The compiled model, or the original model unchanged if compilation
        fails (e.g. unsupported platform or PyTorch version).

    Examples
    --------
    >>> nqs = DenseNQS(num_sites=10, hidden_dims=[64, 32])
    >>> nqs = compile_nqs(nqs, mode="max-autotune")
    """
    try:
        compiled = torch.compile(model, mode=mode)
        logger.info("Successfully compiled NQS model with mode='%s'.", mode)
        return compiled  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch.compile failed (%s). Returning uncompiled model.", exc)
        return model


# ---------------------------------------------------------------------------
# Helper: build a stack of Linear + activation layers
# ---------------------------------------------------------------------------


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU(),
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build a simple MLP as an ``nn.Sequential``.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector.
    hidden_dims : list of int
        Sizes of each hidden layer.
    output_dim : int
        Size of the output layer.
    activation : nn.Module, optional
        Activation function applied after each hidden layer
        (default :class:`~torch.nn.ReLU`).
    output_activation : nn.Module or None, optional
        Optional activation applied after the output layer.

    Returns
    -------
    nn.Sequential
        The assembled MLP.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation)
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# DenseNQS
# ---------------------------------------------------------------------------


class DenseNQS(NeuralQuantumState):
    """Fully connected feedforward neural quantum state.

    The amplitude network maps a configuration vector to a scalar
    log-amplitude via a stack of Linear + ReLU layers, followed by a
    final Linear + Tanh layer whose output is scaled by a learnable
    ``log_amp_scale`` parameter.

    If ``complex_output`` is ``True``, a separate phase network of the
    same depth produces the wavefunction phase in ``(-pi, pi)``.

    Parameters
    ----------
    num_sites : int
        Number of lattice / orbital sites.
    hidden_dims : list of int, optional
        Hidden-layer sizes for the amplitude (and phase) networks
        (default ``[128, 64]``).
    complex_output : bool, optional
        Whether to include a phase network (default ``False``).

    Attributes
    ----------
    amplitude_net : nn.Sequential
        The amplitude MLP (output before scaling).
    log_amp_scale : nn.Parameter
        Learnable scalar that multiplies the Tanh output.
    phase_net : nn.Sequential or None
        Phase MLP when ``complex_output is True``, else ``None``.

    Examples
    --------
    >>> nqs = DenseNQS(num_sites=10, hidden_dims=[64, 32])
    >>> x = torch.randint(0, 2, (8, 10)).float()
    >>> log_amp = nqs.log_amplitude(x)  # shape (8,)
    """

    def __init__(
        self,
        num_sites: int,
        hidden_dims: list[int] | None = None,
        complex_output: bool = False,
    ) -> None:
        super().__init__(
            num_sites=num_sites,
            local_dim=2,
            complex_output=complex_output,
        )
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Amplitude network: Input -> [Linear+ReLU]... -> Linear -> Tanh
        self.amplitude_net: nn.Sequential = _build_mlp(
            input_dim=num_sites,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=nn.ReLU(),
            output_activation=nn.Tanh(),
        )

        # Learnable scale for the log-amplitude
        self.log_amp_scale: nn.Parameter = nn.Parameter(torch.tensor(1.0))

        # Phase network (optional)
        self.phase_net: nn.Sequential | None = None
        if complex_output:
            self.phase_net = _build_mlp(
                input_dim=num_sites,
                hidden_dims=hidden_dims,
                output_dim=1,
                activation=nn.ReLU(),
                output_activation=nn.Tanh(),  # output in (-1, 1)
            )

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-amplitude ln|psi(x)|.

        The raw amplitude network output (bounded in ``(-1, 1)`` by Tanh)
        is multiplied by the learnable ``log_amp_scale``.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Log-amplitudes, shape ``(batch,)``.
        """
        x = self.encode_configuration(x)
        raw = self.amplitude_net(x).squeeze(-1)  # (batch,)
        return raw * self.log_amp_scale

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the wavefunction phase.

        Returns zeros for real-valued NQS.  For complex NQS the phase
        network output (in ``(-1, 1)``) is scaled by ``pi`` so the phase
        lies in ``(-pi, pi)``.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Phases in radians, shape ``(batch,)``.
        """
        if self.phase_net is None:
            return torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        x = self.encode_configuration(x)
        raw = self.phase_net(x).squeeze(-1)  # (batch,)
        return raw * torch.pi


# ---------------------------------------------------------------------------
# SignedDenseNQS
# ---------------------------------------------------------------------------


class SignedDenseNQS(NeuralQuantumState):
    """Dense NQS with explicit sign structure.

    Uses a shared feature extractor whose output feeds into two heads:

    * **Amplitude head** --- produces the log-amplitude via Softplus to
      ensure non-negative output.
    * **Sign head** --- produces a logit whose sigmoid is thresholded at
      0.5 to yield a phase of either 0 (positive) or pi (negative).

    Feature caching avoids redundant computation when :meth:`log_amplitude`
    and :meth:`phase` are called on the same input within one evaluation.

    Parameters
    ----------
    num_sites : int
        Number of lattice / orbital sites.
    hidden_dims : list of int, optional
        Hidden-layer sizes for the shared feature extractor
        (default ``[128, 64]``).

    Attributes
    ----------
    feature_net : nn.Sequential
        Shared feature extractor.
    amplitude_head : nn.Sequential
        Maps features to log-amplitude (Softplus output).
    sign_head : nn.Linear
        Maps features to a sign logit.

    Examples
    --------
    >>> nqs = SignedDenseNQS(num_sites=10)
    >>> x = torch.randint(0, 2, (8, 10)).float()
    >>> log_amp = nqs.log_amplitude(x)  # shape (8,)
    >>> phi = nqs.phase(x)              # shape (8,), values in {0, pi}
    """

    def __init__(
        self,
        num_sites: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__(
            num_sites=num_sites,
            local_dim=2,
            complex_output=True,
        )
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Shared feature extractor
        self.feature_net: nn.Sequential = _build_mlp(
            input_dim=num_sites,
            hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else [],
            output_dim=hidden_dims[-1],
            activation=nn.ReLU(),
            output_activation=nn.ReLU(),
        )

        feature_dim = hidden_dims[-1]

        # Amplitude head: features -> Softplus -> scalar
        self.amplitude_head: nn.Sequential = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Softplus(),
        )

        # Sign head: features -> logit
        self.sign_head: nn.Linear = nn.Linear(feature_dim, 1)

        # Feature cache
        self._cached_input_id: int | None = None
        self._cached_features: torch.Tensor | None = None

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute (or retrieve cached) shared features.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Feature vectors, shape ``(batch, feature_dim)``.
        """
        input_id = id(x)
        if self._cached_input_id == input_id and self._cached_features is not None:
            return self._cached_features

        encoded = self.encode_configuration(x)
        features = self.feature_net(encoded)
        self._cached_input_id = input_id
        self._cached_features = features
        return features

    def clear_feature_cache(self) -> None:
        """Clear the feature cache.

        Call this between training steps or whenever the input batch
        changes to avoid stale cached values.
        """
        self._cached_input_id = None
        self._cached_features = None

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-amplitude from the amplitude head.

        The Softplus activation ensures the raw amplitude is non-negative;
        the log-amplitude is the logarithm of that value.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Log-amplitudes, shape ``(batch,)``.
        """
        features = self._get_features(x)
        # Softplus output is already positive; take log for log-amplitude
        amp = self.amplitude_head(features).squeeze(-1)  # (batch,)
        return torch.log(amp + 1e-12)

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the sign-derived phase.

        The sign logit is passed through a sigmoid.  Values above 0.5
        correspond to a positive sign (phase = 0); values below 0.5
        correspond to a negative sign (phase = pi).

        During training, a soft interpolation is used for gradient flow:
        ``phase = pi * (1 - sigmoid(sign_logit))``.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Phase values, shape ``(batch,)``.  During training these are
            continuous in ``(0, pi)``; at eval they snap to ``{0, pi}``.
        """
        features = self._get_features(x)
        sign_logit = self.sign_head(features).squeeze(-1)  # (batch,)

        if self.training:
            # Soft version for gradient flow
            return torch.pi * (1.0 - torch.sigmoid(sign_logit))

        # Hard threshold at eval time
        positive = torch.sigmoid(sign_logit) >= 0.5
        return torch.where(
            positive,
            torch.zeros_like(sign_logit),
            torch.full_like(sign_logit, torch.pi),
        )
