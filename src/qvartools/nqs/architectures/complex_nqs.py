"""
complex_nqs --- Complex-valued dense NQS architecture
=====================================================

Provides :class:`ComplexNQS`, a dense NQS with shared feature extractor
and separate amplitude/phase heads.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from qvartools.nqs.neural_state import NeuralQuantumState
from qvartools.nqs.architectures.rbm import RBMQuantumState  # noqa: F401

__all__ = [
    "ComplexNQS",
    "RBMQuantumState",
]


# ---------------------------------------------------------------------------
# ComplexNQS
# ---------------------------------------------------------------------------


class ComplexNQS(NeuralQuantumState):
    """Complex-valued NQS with shared feature extractor.

    A shared MLP backbone produces feature vectors that are fed into two
    independent heads:

    * **Amplitude head** --- maps features to a scalar log-amplitude.
    * **Phase head** --- maps features to a phase in ``(-pi, pi)``.

    Feature caching avoids redundant computation when :meth:`log_amplitude`
    and :meth:`phase` are called sequentially on the same input tensor.

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
        Shared feature MLP.
    amplitude_head : nn.Linear
        Linear projection from features to scalar log-amplitude.
    phase_head : nn.Sequential
        Maps features to phase via Linear + Tanh (scaled by pi).

    Examples
    --------
    >>> nqs = ComplexNQS(num_sites=10, hidden_dims=[64, 32])
    >>> x = torch.randint(0, 2, (8, 10)).float()
    >>> log_amp = nqs.log_amplitude(x)  # shape (8,)
    >>> phi = nqs.phase(x)              # shape (8,), in (-pi, pi)
    """

    def __init__(
        self,
        num_sites: int,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            num_sites=num_sites,
            local_dim=2,
            complex_output=True,
        )
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Build shared feature extractor
        layers: list[nn.Module] = []
        prev_dim = num_sites
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.feature_net: nn.Sequential = nn.Sequential(*layers)

        feature_dim = hidden_dims[-1]

        # Amplitude head: single linear -> scalar
        self.amplitude_head: nn.Linear = nn.Linear(feature_dim, 1)

        # Phase head: linear -> tanh (output in (-1,1), scaled by pi)
        self.phase_head: nn.Sequential = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Tanh(),
        )

        # Feature cache
        self._cached_input_id: Optional[int] = None
        self._cached_features: Optional[torch.Tensor] = None

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

        Call this between training steps or when the input batch changes
        to avoid stale cached values.
        """
        self._cached_input_id = None
        self._cached_features = None

    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-amplitude from shared features.

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
        return self.amplitude_head(features).squeeze(-1)

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Compute phase from shared features.

        The Tanh output (in ``(-1, 1)``) is scaled by ``pi`` to produce
        a phase in ``(-pi, pi)``.

        Parameters
        ----------
        x : torch.Tensor
            Batch of configurations, shape ``(batch, num_sites)``.

        Returns
        -------
        torch.Tensor
            Phases in radians, shape ``(batch,)``.
        """
        features = self._get_features(x)
        raw = self.phase_head(features).squeeze(-1)
        return raw * torch.pi
