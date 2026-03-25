"""
transformer_nf_sampler --- Transformer-based normalizing-flow sampler
=====================================================================

Implements :class:`TransformerNFSampler`, a drop-in replacement for
:class:`~qvartools.samplers.nf_sampler.NFSampler` that uses a fully
autoregressive transformer instead of the product-of-marginals flow.
Each orbital probability ``P(sigma_i | sigma_{<i})`` is conditioned on
**all** previous decisions via causal self-attention, and the beta spin
channel cross-attends to the full alpha configuration.

Architecture references:

* Psiformer (von Glehn et al., 2023) -- self-attention for quantum
  chemistry.
* Autoregressive NQS (Sharir et al., 2020) -- autoregressive
  factorisation for wavefunctions.
* Barrett et al. (2022) -- autoregressive wavefunctions for *ab initio*
  quantum chemistry.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer
from qvartools.samplers.sampler import Sampler, SamplerResult

__all__ = [
    "TransformerSamplerConfig",
    "TransformerNFSampler",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter auto-scaling
# ---------------------------------------------------------------------------


def _auto_scale_transformer(n_orbitals: int) -> Dict[str, int]:
    """Choose transformer hyper-parameters based on system size.

    Parameters
    ----------
    n_orbitals : int
        Number of spatial orbitals.

    Returns
    -------
    dict
        Keys ``embed_dim``, ``n_heads``, ``n_layers``.
    """
    if n_orbitals <= 10:
        return {"embed_dim": 64, "n_heads": 4, "n_layers": 4}
    if n_orbitals <= 15:
        return {"embed_dim": 128, "n_heads": 4, "n_layers": 4}
    if n_orbitals <= 20:
        return {"embed_dim": 128, "n_heads": 8, "n_layers": 6}
    if n_orbitals <= 26:
        return {"embed_dim": 192, "n_heads": 8, "n_layers": 6}
    return {"embed_dim": 256, "n_heads": 8, "n_layers": 8}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformerSamplerConfig:
    """Configuration for :class:`TransformerNFSampler`.

    Parameters
    ----------
    n_epochs : int
        Number of training epochs (default ``400``).
    samples_per_epoch : int
        Samples drawn per training epoch (default ``512``).
    flow_lr : float
        Learning rate for the flow / transformer (default ``3e-4``).
    nqs_lr : float
        Learning rate for the NQS critic (default ``1e-3``).
    initial_temperature : float
        Gumbel temperature at training start (default ``1.0``).
    final_temperature : float
        Gumbel temperature at training end (default ``0.3``).
    embed_dim : int or None
        Embedding dimension (``None`` = auto-scale).
    n_heads : int or None
        Number of attention heads (``None`` = auto-scale).
    n_layers : int or None
        Number of transformer layers (``None`` = auto-scale).
    nqs_embed_dim : int or None
        NQS embedding dimension (``None`` = match flow).
    nqs_n_heads : int or None
        NQS attention heads (``None`` = match flow).
    nqs_n_layers : int or None
        NQS transformer layers (``None`` = ``n_layers - 1``, min 2).
    """

    n_epochs: int = 400
    samples_per_epoch: int = 512
    flow_lr: float = 3e-4
    nqs_lr: float = 1e-3
    initial_temperature: float = 1.0
    final_temperature: float = 0.3

    # Transformer architecture (None = auto-scale)
    embed_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_layers: Optional[int] = None

    # NQS architecture
    nqs_embed_dim: Optional[int] = None
    nqs_n_heads: Optional[int] = None
    nqs_n_layers: Optional[int] = None


# ---------------------------------------------------------------------------
# Flow wrapper
# ---------------------------------------------------------------------------


class _TransformerFlowWrapper(nn.Module):
    """Adapter making :class:`AutoregressiveTransformer` compatible
    with the :class:`PhysicsGuidedFlowTrainer` interface.

    Parameters
    ----------
    transformer : AutoregressiveTransformer
        The underlying autoregressive model.
    """

    def __init__(self, transformer: AutoregressiveTransformer) -> None:
        super().__init__()
        self.flow: AutoregressiveTransformer = transformer
        self.num_sites: int = transformer.n_qubits
        self.temperature: float = 1.0

    def sample(
        self, n_samples: int, hard: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample and return ``(log_probs, unique_configs)``.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        hard : bool
            Whether to use hard (discrete) sampling.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(log_probs, unique_configs)``.
        """
        configs, log_probs = self.flow.sample(
            n_samples, hard=hard, temperature=self.temperature
        )
        unique_configs = torch.unique(configs.long(), dim=0)
        return log_probs, unique_configs

    def log_prob(self, config: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of a batch of configurations.

        Parameters
        ----------
        config : torch.Tensor
            Configurations, shape ``(batch, n_qubits)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities, shape ``(batch,)``.
        """
        return self.flow.log_prob(config)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class TransformerNFSampler(Sampler):
    """Transformer-based normalizing-flow sampler.

    Drop-in replacement for :class:`~qvartools.samplers.nf_sampler.NFSampler`
    with a fully autoregressive transformer architecture that captures
    inter-orbital correlations.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian providing ``num_sites``, ``n_orbitals``,
        ``n_alpha``, and ``n_beta``.
    config : TransformerSamplerConfig or None, optional
        Sampler / training configuration.  Uses defaults when ``None``.
    device : str, optional
        Torch device (default ``"cpu"``).

    Examples
    --------
    >>> sampler = TransformerNFSampler(hamiltonian)
    >>> sampler.train()  # physics-guided training
    >>> result = sampler.sample(2048)
    >>> result.configs.shape[1]  # == 2 * n_orbitals
    10
    """

    def __init__(
        self,
        hamiltonian,
        config: Optional[TransformerSamplerConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.hamiltonian = hamiltonian
        self.config: TransformerSamplerConfig = (
            config or TransformerSamplerConfig()
        )
        self.device: str = device

        n_sites: int = hamiltonian.num_sites
        n_orbitals: int = hamiltonian.n_orbitals
        n_alpha: int = hamiltonian.n_alpha
        n_beta: int = hamiltonian.n_beta

        # Auto-scale architecture if not specified
        auto = _auto_scale_transformer(n_orbitals)
        embed_dim = self.config.embed_dim or auto["embed_dim"]
        n_heads = self.config.n_heads or auto["n_heads"]
        n_layers = self.config.n_layers or auto["n_layers"]

        # Build autoregressive transformer
        self.transformer: AutoregressiveTransformer = (
            AutoregressiveTransformer(
                n_orbitals=n_orbitals,
                n_alpha=n_alpha,
                n_beta=n_beta,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_layers=n_layers,
            ).to(device)
        )

        # Wrap for trainer compatibility
        self.flow: _TransformerFlowWrapper = _TransformerFlowWrapper(
            self.transformer
        ).to(device)

        # Build NQS critic
        nqs_embed = self.config.nqs_embed_dim or embed_dim
        nqs_heads = self.config.nqs_n_heads or n_heads
        nqs_layers = self.config.nqs_n_layers or max(n_layers - 1, 2)

        self.nqs: nn.Module = self._build_nqs(
            n_sites, nqs_embed, nqs_heads, nqs_layers
        )

        self._trained: bool = False

    # ------------------------------------------------------------------
    # NQS construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_nqs(
        n_sites: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
    ) -> nn.Module:
        """Instantiate the NQS critic network.

        Attempts to use ``TransformerNQS`` from ``qvartools.nqs`` and
        falls back to a :class:`DenseNQS` if unavailable.

        Parameters
        ----------
        n_sites : int
            Number of qubit sites.
        embed_dim : int
            Embedding dimension.
        n_heads : int
            Attention head count.
        n_layers : int
            Number of layers.

        Returns
        -------
        nn.Module
            The NQS model.
        """
        # Try transformer NQS first
        try:
            from qvartools.nqs.transformer.autoregressive import TransformerNQS  # noqa: WPS433

            return TransformerNQS(
                num_sites=n_sites,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_layers=n_layers,
            )
        except (ImportError, AttributeError):
            pass

        # Fallback to dense NQS
        from qvartools.nqs.architectures.dense import DenseNQS

        return DenseNQS(num_sites=n_sites, hidden_dim=embed_dim)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """Train the transformer and NQS via physics-guided training.

        Parameters
        ----------
        verbose : bool, optional
            Whether to log training progress (default ``True``).

        Returns
        -------
        dict
            Training history from the trainer.
        """
        from qvartools.flows.training.physics_guided_training import (
            PhysicsGuidedConfig,
            PhysicsGuidedFlowTrainer,
        )

        cfg = self.config
        train_config = PhysicsGuidedConfig(
            samples_per_batch=cfg.samples_per_epoch,
            num_epochs=cfg.n_epochs,
            flow_lr=cfg.flow_lr,
            nqs_lr=cfg.nqs_lr,
            initial_temperature=cfg.initial_temperature,
            final_temperature=cfg.final_temperature,
        )

        trainer = PhysicsGuidedFlowTrainer(
            flow=self.flow,
            nqs=self.nqs,
            hamiltonian=self.hamiltonian,
            config=train_config,
            device=self.device,
        )

        history = trainer.train()
        self._trained = True
        return history

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples: int) -> SamplerResult:
        """Sample configurations from the trained transformer.

        If the model has not been trained yet, :meth:`train` is called
        automatically.

        Parameters
        ----------
        n_samples : int
            Number of raw samples to draw (unique configurations are
            returned).

        Returns
        -------
        SamplerResult
            Unique configurations with log-probabilities and metadata.

        Raises
        ------
        ValueError
            If ``n_samples < 1``.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        if not self._trained:
            self.train()

        t_start = time.perf_counter()

        with torch.no_grad():
            self.flow.temperature = self.config.final_temperature
            configs, log_probs = self.transformer.sample(
                n_samples, temperature=self.config.final_temperature
            )
            configs = configs.long()

            # Deduplicate
            unique_configs = torch.unique(configs, dim=0)

            # Recompute log-probs for unique configurations
            unique_log_probs = self.transformer.log_prob(unique_configs)

        wall_time = time.perf_counter() - t_start

        # Build bitstring counts from raw (non-deduplicated) samples
        bitstrings = [
            "".join(str(int(b)) for b in row) for row in configs.cpu().int()
        ]
        counts: Dict[str, int] = dict(Counter(bitstrings))

        metadata: Dict[str, Any] = {
            "n_raw_samples": n_samples,
            "n_unique": len(unique_configs),
            "unique_ratio": len(unique_configs) / max(n_samples, 1),
            "architecture": "transformer",
            "sampler_type": "TransformerNF",
        }

        logger.info(
            "TransformerNFSampler: drew %d samples (%d unique) in %.3fs",
            n_samples,
            len(unique_configs),
            wall_time,
        )

        return SamplerResult(
            configs=unique_configs,
            counts=counts,
            log_probs=unique_log_probs,
            wall_time=wall_time,
            metadata=metadata,
        )
