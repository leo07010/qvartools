"""
physics_guided_training --- Mixed-objective flow + NQS trainer
==============================================================

Implements :class:`PhysicsGuidedFlowTrainer`, a training orchestrator
that jointly optimises a normalizing flow and a neural quantum state
using a combination of three loss terms:

* **Teacher loss** --- trains the flow to match the NQS distribution by
  maximising ``log p_flow(x)`` weighted by the NQS probability.
* **Physics loss** --- minimises the variational energy via the local
  energy estimator ``E_loc(x) = sum_x' H_{x,x'} psi(x') / psi(x)``.
* **Entropy loss** --- encourages exploration by maximising the entropy
  of the flow distribution.

The training loop includes temperature annealing for particle-conserving
flows, essential-configuration injection (Hartree--Fock + singles +
doubles), and convergence detection based on the unique-configuration
ratio.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.flows.training.loss_functions import (
    ConnectionCache,
    compute_entropy_loss,
    compute_physics_loss,
    compute_teacher_loss,
)

__all__ = [
    "PhysicsGuidedConfig",
    "PhysicsGuidedFlowTrainer",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsGuidedConfig:
    """Hyperparameters for :class:`PhysicsGuidedFlowTrainer`.

    All fields have sensible defaults for molecular-scale problems.
    The class is frozen (immutable) to prevent accidental mutation
    during training.

    Parameters
    ----------
    samples_per_batch : int
        Number of flow samples per mini-batch (default ``500``).
    num_batches : int
        Number of mini-batches per epoch (default ``10``).
    num_epochs : int
        Maximum number of training epochs (default ``200``).
    min_epochs : int
        Minimum epochs before convergence checks activate (default ``50``).
    convergence_threshold : float
        Training stops when the unique-configuration ratio changes by
        less than this amount over consecutive epochs (default ``0.01``).
    flow_lr : float
        Learning rate for the flow optimiser (default ``1e-3``).
    nqs_lr : float
        Learning rate for the NQS optimiser (default ``1e-3``).
    teacher_weight : float
        Weight of the teacher (KL) loss (default ``1.0``).
    physics_weight : float
        Weight of the variational energy loss (default ``0.0``).
    entropy_weight : float
        Weight of the entropy regularisation loss (default ``0.0``).
    use_energy_baseline : bool
        Whether to subtract a running baseline from the energy for
        variance reduction (default ``True``).
    ema_decay : float
        Exponential moving average decay for the energy baseline
        (default ``0.99``).
    use_connection_cache : bool
        Whether to cache Hamiltonian connections for repeated configs
        (default ``True``).
    max_cache_size : int
        Maximum number of entries in the connection cache
        (default ``100000``).
    initial_temperature : float
        Starting temperature for flow annealing (default ``2.0``).
    final_temperature : float
        Final temperature after annealing (default ``0.1``).
    temperature_decay_epochs : int
        Number of epochs over which to anneal temperature
        (default ``100``).
    inject_essential_configs : bool
        Whether to inject Hartree--Fock and nearby configurations into
        the basis (default ``True``).
    include_singles_in_basis : bool
        Whether to include single excitations in the essential basis
        (default ``True``).
    include_doubles_in_basis : bool
        Whether to include double excitations in the essential basis
        (default ``True``).
    device : str
        Torch device for training (default ``"cpu"``).
    """

    samples_per_batch: int = 500
    num_batches: int = 10
    num_epochs: int = 200
    min_epochs: int = 50
    convergence_threshold: float = 0.01
    flow_lr: float = 1e-3
    nqs_lr: float = 1e-3
    teacher_weight: float = 1.0
    physics_weight: float = 0.0
    entropy_weight: float = 0.0
    use_energy_baseline: bool = True
    ema_decay: float = 0.99
    use_connection_cache: bool = True
    max_cache_size: int = 100000
    initial_temperature: float = 2.0
    final_temperature: float = 0.1
    temperature_decay_epochs: int = 100
    inject_essential_configs: bool = True
    include_singles_in_basis: bool = True
    include_doubles_in_basis: bool = True
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Essential configuration generators
# ---------------------------------------------------------------------------


def _generate_hf_config(
    n_orbitals: int, n_alpha: int, n_beta: int
) -> torch.Tensor:
    """Generate the Hartree--Fock reference configuration.

    The HF configuration occupies the lowest-energy orbitals: the first
    ``n_alpha`` alpha orbitals and the first ``n_beta`` beta orbitals.

    Parameters
    ----------
    n_orbitals : int
        Number of spatial orbitals.
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.

    Returns
    -------
    torch.Tensor
        Binary HF configuration, shape ``(2 * n_orbitals,)``.
    """
    config = torch.zeros(2 * n_orbitals, dtype=torch.float32)
    config[:n_alpha] = 1.0
    config[n_orbitals : n_orbitals + n_beta] = 1.0
    return config


def _generate_single_excitations(
    hf_config: torch.Tensor,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
) -> torch.Tensor:
    """Generate all single excitations from the HF configuration.

    Parameters
    ----------
    hf_config : torch.Tensor
        The Hartree--Fock reference, shape ``(2 * n_orbitals,)``.
    n_orbitals : int
        Number of spatial orbitals.
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.

    Returns
    -------
    torch.Tensor
        Single-excitation configurations, shape ``(n_singles, 2 * n_orbitals)``.
    """
    singles: list[torch.Tensor] = []

    # Alpha single excitations: i (occupied) -> a (virtual)
    for i in range(n_alpha):
        for a in range(n_alpha, n_orbitals):
            config = hf_config.clone()
            config[i] = 0.0
            config[a] = 1.0
            singles.append(config)

    # Beta single excitations
    for i in range(n_orbitals, n_orbitals + n_beta):
        for a in range(n_orbitals + n_beta, 2 * n_orbitals):
            config = hf_config.clone()
            config[i] = 0.0
            config[a] = 1.0
            singles.append(config)

    if not singles:
        return torch.empty(0, 2 * n_orbitals, dtype=torch.float32)
    return torch.stack(singles)


def _generate_double_excitations(
    hf_config: torch.Tensor,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
) -> torch.Tensor:
    """Generate all double excitations from the HF configuration.

    Includes alpha-alpha, beta-beta, and alpha-beta double excitations.

    Parameters
    ----------
    hf_config : torch.Tensor
        The Hartree--Fock reference, shape ``(2 * n_orbitals,)``.
    n_orbitals : int
        Number of spatial orbitals.
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.

    Returns
    -------
    torch.Tensor
        Double-excitation configurations, shape ``(n_doubles, 2 * n_orbitals)``.
    """
    doubles: list[torch.Tensor] = []

    alpha_occ = list(range(n_alpha))
    alpha_vir = list(range(n_alpha, n_orbitals))
    beta_occ = list(range(n_orbitals, n_orbitals + n_beta))
    beta_vir = list(range(n_orbitals + n_beta, 2 * n_orbitals))

    # Alpha-alpha doubles
    for i_idx in range(len(alpha_occ)):
        for j_idx in range(i_idx + 1, len(alpha_occ)):
            i, j = alpha_occ[i_idx], alpha_occ[j_idx]
            for a_idx in range(len(alpha_vir)):
                for b_idx in range(a_idx + 1, len(alpha_vir)):
                    a, b = alpha_vir[a_idx], alpha_vir[b_idx]
                    config = hf_config.clone()
                    config[i] = 0.0
                    config[j] = 0.0
                    config[a] = 1.0
                    config[b] = 1.0
                    doubles.append(config)

    # Beta-beta doubles
    for i_idx in range(len(beta_occ)):
        for j_idx in range(i_idx + 1, len(beta_occ)):
            i, j = beta_occ[i_idx], beta_occ[j_idx]
            for a_idx in range(len(beta_vir)):
                for b_idx in range(a_idx + 1, len(beta_vir)):
                    a, b = beta_vir[a_idx], beta_vir[b_idx]
                    config = hf_config.clone()
                    config[i] = 0.0
                    config[j] = 0.0
                    config[a] = 1.0
                    config[b] = 1.0
                    doubles.append(config)

    # Alpha-beta doubles
    for i in alpha_occ:
        for a in alpha_vir:
            for j in beta_occ:
                for b in beta_vir:
                    config = hf_config.clone()
                    config[i] = 0.0
                    config[a] = 1.0
                    config[j] = 0.0
                    config[b] = 1.0
                    doubles.append(config)

    if not doubles:
        return torch.empty(0, 2 * n_orbitals, dtype=torch.float32)
    return torch.stack(doubles)


# ---------------------------------------------------------------------------
# PhysicsGuidedFlowTrainer
# ---------------------------------------------------------------------------


class PhysicsGuidedFlowTrainer:
    """Mixed-objective trainer for joint flow + NQS optimisation.

    Combines three loss terms with configurable weights:

    * **Teacher loss**: ``-sum_x p_nqs(x) * log p_flow(x)`` ---
      trains the flow to reproduce the NQS distribution.
    * **Physics loss**: variational energy
      ``E = sum_x |psi(x)|^2 * E_loc(x) / Z`` where
      ``E_loc(x) = sum_{x'} H_{x,x'} * psi(x') / psi(x)`` ---
      minimises the ground-state energy estimate.
    * **Entropy loss**: ``-H[p_flow]`` (negative entropy) ---
      prevents mode collapse by encouraging distribution spread.

    The trainer also:

    * Accumulates unique configurations into a growing basis set.
    * Anneals the flow temperature over early epochs.
    * Optionally injects essential (HF + singles + doubles)
      configurations into the basis.
    * Caches Hamiltonian connections for efficiency.

    Parameters
    ----------
    flow : nn.Module
        The normalizing flow sampler.  Must implement ``sample(batch_size)``
        returning ``(all_configs, unique_configs)``.
    nqs : nn.Module
        The neural quantum state.  Must implement ``log_amplitude(x)``
        returning log-amplitudes of shape ``(batch,)``.
    hamiltonian : Hamiltonian
        The Hamiltonian operator.
    config : PhysicsGuidedConfig
        Training hyperparameters.
    device : str, optional
        Torch device override (default uses ``config.device``).

    Attributes
    ----------
    flow : nn.Module
        The flow model.
    nqs : nn.Module
        The NQS model.
    hamiltonian : Hamiltonian
        The Hamiltonian.
    config : PhysicsGuidedConfig
        Training configuration.
    device : torch.device
        Active device.
    accumulated_basis : torch.Tensor or None
        Growing set of unique configurations seen during training.
    flow_optimizer : torch.optim.Adam
        Optimiser for the flow parameters.
    nqs_optimizer : torch.optim.Adam
        Optimiser for the NQS parameters.
    energy_baseline : float
        Running EMA baseline for variance reduction.
    connection_cache : ConnectionCache or None
        Cache for Hamiltonian connections.
    """

    def __init__(
        self,
        flow: nn.Module,
        nqs: nn.Module,
        hamiltonian: Hamiltonian,
        config: PhysicsGuidedConfig,
        device: str = "cpu",
    ) -> None:
        effective_device = device if device != "cpu" else config.device
        self.device: torch.device = torch.device(effective_device)

        self.flow: nn.Module = flow.to(self.device)
        self.nqs: nn.Module = nqs.to(self.device)
        self.hamiltonian: Hamiltonian = hamiltonian
        self.config: PhysicsGuidedConfig = config

        self.accumulated_basis: Optional[torch.Tensor] = None

        self.flow_optimizer: torch.optim.Adam = torch.optim.Adam(
            flow.parameters(), lr=config.flow_lr
        )
        self.nqs_optimizer: torch.optim.Adam = torch.optim.Adam(
            nqs.parameters(), lr=config.nqs_lr
        )

        self.energy_baseline: float = 0.0
        self._baseline_initialized: bool = False

        self.connection_cache: Optional[ConnectionCache] = None
        if config.use_connection_cache:
            self.connection_cache = ConnectionCache(
                max_size=config.max_cache_size
            )

        # Inject essential configurations if requested
        if config.inject_essential_configs:
            self._inject_essential_configs()

    def _inject_essential_configs(self) -> None:
        """Inject HF + singles + doubles into the accumulated basis."""
        num_sites = self.hamiltonian.num_sites
        n_orbitals = num_sites // 2

        # Attempt to infer n_alpha, n_beta from the flow
        n_alpha = getattr(self.flow, "n_alpha", None)
        n_beta = getattr(self.flow, "n_beta", None)

        if n_alpha is None or n_beta is None:
            logger.warning(
                "Cannot inject essential configs: flow does not expose "
                "n_alpha and n_beta attributes."
            )
            return

        essential: list[torch.Tensor] = []

        # Hartree-Fock reference
        hf = _generate_hf_config(n_orbitals, n_alpha, n_beta)
        essential.append(hf)

        # Single excitations
        if self.config.include_singles_in_basis:
            singles = _generate_single_excitations(
                hf, n_orbitals, n_alpha, n_beta
            )
            if singles.shape[0] > 0:
                essential.append(singles)

        # Double excitations
        if self.config.include_doubles_in_basis:
            doubles = _generate_double_excitations(
                hf, n_orbitals, n_alpha, n_beta
            )
            if doubles.shape[0] > 0:
                essential.append(doubles)

        if essential:
            all_essential = torch.cat(
                [e.unsqueeze(0) if e.ndim == 1 else e for e in essential],
                dim=0,
            ).to(self.device)
            self._accumulate_configs(all_essential)
            logger.info(
                "Injected %d essential configurations into basis.",
                all_essential.shape[0],
            )

    def _accumulate_configs(self, new_configs: torch.Tensor) -> None:
        """Add new unique configurations to the accumulated basis.

        Parameters
        ----------
        new_configs : torch.Tensor
            New configurations, shape ``(n, num_sites)``.
        """
        if new_configs.numel() == 0:
            return

        new_configs = new_configs.to(self.device)

        if self.accumulated_basis is None:
            self.accumulated_basis = torch.unique(new_configs, dim=0)
        else:
            combined = torch.cat(
                [self.accumulated_basis, new_configs], dim=0
            )
            self.accumulated_basis = torch.unique(combined, dim=0)

    def _get_temperature(self, epoch: int) -> float:
        """Compute the annealed temperature for the current epoch.

        Uses exponential decay from ``initial_temperature`` to
        ``final_temperature`` over ``temperature_decay_epochs``.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).

        Returns
        -------
        float
            Temperature for the current epoch.
        """
        cfg = self.config
        if epoch >= cfg.temperature_decay_epochs:
            return cfg.final_temperature

        decay_rate = math.log(cfg.initial_temperature / cfg.final_temperature)
        progress = epoch / max(cfg.temperature_decay_epochs, 1)
        return cfg.initial_temperature * math.exp(-decay_rate * progress)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute a single training epoch.

        Samples configurations from the flow, computes the combined loss,
        updates both flow and NQS parameters, and accumulates unique
        configurations into the basis.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).

        Returns
        -------
        dict
            Epoch metrics with keys:

            - ``"teacher_loss"`` : float
            - ``"physics_loss"`` : float
            - ``"entropy_loss"`` : float
            - ``"total_loss"`` : float
            - ``"mean_energy"`` : float
            - ``"unique_ratio"`` : float
            - ``"basis_size"`` : int
            - ``"temperature"`` : float
        """
        cfg = self.config
        self.flow.train()
        self.nqs.train()

        # Temperature annealing
        temperature = self._get_temperature(epoch)
        if hasattr(self.flow, "set_temperature"):
            self.flow.set_temperature(temperature)

        epoch_teacher = 0.0
        epoch_physics = 0.0
        epoch_entropy = 0.0
        epoch_total = 0.0
        epoch_energy = 0.0
        total_samples = 0
        total_unique = 0

        for _ in range(cfg.num_batches):
            # Sample from flow
            sample_result = self.flow.sample(cfg.samples_per_batch)
            if len(sample_result) == 2:
                all_configs, unique_configs = sample_result
            else:
                # Handle flows that return extra outputs
                all_configs, unique_configs = sample_result[0], sample_result[1]

            all_configs = all_configs.to(self.device)
            unique_configs = unique_configs.to(self.device)

            total_samples += all_configs.shape[0]
            total_unique += unique_configs.shape[0]

            # Accumulate basis
            self._accumulate_configs(unique_configs)

            # Compute flow log-probabilities for loss terms
            # Use continuous log-prob if available, else approximate
            if hasattr(self.flow, "log_prob_continuous"):
                # Map discrete configs to continuous space center points
                y_approx = 2.0 * all_configs - 1.0  # {0,1} -> {-1,+1}
                log_probs_flow = self.flow.log_prob_continuous(y_approx)
            else:
                # Fallback: use uniform log-prob (no teacher signal)
                log_probs_flow = torch.zeros(
                    all_configs.shape[0], device=self.device
                )

            # Compute losses
            loss = torch.tensor(0.0, device=self.device)

            teacher_loss_val = 0.0
            if cfg.teacher_weight > 0.0:
                t_loss = compute_teacher_loss(
                    all_configs, log_probs_flow, self.nqs
                )
                loss = loss + cfg.teacher_weight * t_loss
                teacher_loss_val = float(t_loss.detach())

            physics_loss_val = 0.0
            energy_val = 0.0
            if cfg.physics_weight > 0.0:
                p_loss, energy_val, self.energy_baseline, self._baseline_initialized = (
                    compute_physics_loss(
                        all_configs,
                        self.nqs,
                        self.hamiltonian,
                        self.device,
                        self.energy_baseline,
                        self._baseline_initialized,
                        cfg.use_energy_baseline,
                        cfg.ema_decay,
                        self.connection_cache,
                    )
                )
                loss = loss + cfg.physics_weight * p_loss
                physics_loss_val = float(p_loss.detach())

            entropy_loss_val = 0.0
            if cfg.entropy_weight > 0.0:
                e_loss = compute_entropy_loss(log_probs_flow)
                loss = loss + cfg.entropy_weight * e_loss
                entropy_loss_val = float(e_loss.detach())

            # Backward pass and optimiser steps
            self.flow_optimizer.zero_grad()
            self.nqs_optimizer.zero_grad()

            if loss.requires_grad:
                loss.backward()
                self.flow_optimizer.step()
                self.nqs_optimizer.step()

            epoch_teacher += teacher_loss_val
            epoch_physics += physics_loss_val
            epoch_entropy += entropy_loss_val
            epoch_total += float(loss.detach()) if loss.requires_grad else float(loss)
            epoch_energy += energy_val

        n_batches = max(cfg.num_batches, 1)
        unique_ratio = total_unique / max(total_samples, 1)
        basis_size = (
            self.accumulated_basis.shape[0]
            if self.accumulated_basis is not None
            else 0
        )

        return {
            "teacher_loss": epoch_teacher / n_batches,
            "physics_loss": epoch_physics / n_batches,
            "entropy_loss": epoch_entropy / n_batches,
            "total_loss": epoch_total / n_batches,
            "mean_energy": epoch_energy / n_batches,
            "unique_ratio": unique_ratio,
            "basis_size": basis_size,
            "temperature": temperature,
        }

    def train(self, progress: bool = True) -> Dict[str, list]:
        """Run the full training loop.

        Trains for up to ``config.num_epochs`` epochs, with early
        stopping when the unique-configuration ratio converges (change
        less than ``config.convergence_threshold`` for two consecutive
        epochs after ``config.min_epochs``).

        Parameters
        ----------
        progress : bool, optional
            If ``True``, log epoch-level metrics at INFO level
            (default ``True``).

        Returns
        -------
        dict
            Training history with keys matching the epoch metrics,
            each mapping to a list of per-epoch values:

            - ``"teacher_loss"`` : list of float
            - ``"physics_loss"`` : list of float
            - ``"entropy_loss"`` : list of float
            - ``"total_loss"`` : list of float
            - ``"mean_energy"`` : list of float
            - ``"unique_ratio"`` : list of float
            - ``"basis_size"`` : list of int
            - ``"temperature"`` : list of float
        """
        history: Dict[str, list] = {
            "teacher_loss": [],
            "physics_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "mean_energy": [],
            "unique_ratio": [],
            "basis_size": [],
            "temperature": [],
        }

        prev_unique_ratio = 0.0

        for epoch in range(self.config.num_epochs):
            metrics = self._train_epoch(epoch)

            for key in history:
                history[key].append(metrics[key])

            if progress:
                logger.info(
                    "Epoch %3d | loss=%.4f | energy=%.6f | "
                    "unique_ratio=%.4f | basis=%d | temp=%.3f",
                    epoch,
                    metrics["total_loss"],
                    metrics["mean_energy"],
                    metrics["unique_ratio"],
                    metrics["basis_size"],
                    metrics["temperature"],
                )

            # Convergence check
            if epoch >= self.config.min_epochs:
                delta = abs(metrics["unique_ratio"] - prev_unique_ratio)
                if delta < self.config.convergence_threshold:
                    logger.info(
                        "Converged at epoch %d (unique_ratio delta=%.6f < %.6f).",
                        epoch,
                        delta,
                        self.config.convergence_threshold,
                    )
                    break

            prev_unique_ratio = metrics["unique_ratio"]

        return history
