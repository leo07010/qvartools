"""
loss_functions --- Loss computation for physics-guided flow training
====================================================================

Standalone loss functions and supporting utilities extracted from the
physics-guided training loop:

* :func:`compute_teacher_loss` --- KL-divergence teacher loss.
* :func:`compute_physics_loss` --- Variational energy loss with EMA
  baseline for variance reduction.
* :func:`compute_entropy_loss` --- Negative-entropy regularisation.
* :func:`compute_local_energy` --- Per-configuration local energy via
  the Hamiltonian connections.
* :class:`ConnectionCache` --- LRU-style cache for Hamiltonian
  connection lookups.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qvartools._utils.hashing.connection_cache import ConnectionCache
from qvartools.hamiltonians.hamiltonian import Hamiltonian

__all__ = [
    "ConnectionCache",
    "compute_teacher_loss",
    "compute_physics_loss",
    "compute_entropy_loss",
    "compute_local_energy",
]


# ---------------------------------------------------------------------------
# Local energy computation
# ---------------------------------------------------------------------------


def compute_local_energy(
    configs: torch.Tensor,
    nqs: nn.Module,
    hamiltonian: Hamiltonian,
    device: torch.device,
    connection_cache: ConnectionCache | None = None,
) -> torch.Tensor:
    """Compute the local energy E_loc(x) for each configuration.

    ``E_loc(x) = H_{x,x} + sum_{x' != x} H_{x,x'} * psi(x') / psi(x)``

    Optimised to minimise CPU-GPU transfers and batch all NQS evaluations
    into a single call.

    Parameters
    ----------
    configs : torch.Tensor
        Configurations, shape ``(batch, num_sites)``.
    nqs : nn.Module
        Neural quantum state with a ``log_amplitude(x)`` method.
    hamiltonian : Hamiltonian
        The Hamiltonian operator.
    device : torch.device
        Torch device for computation.
    connection_cache : ConnectionCache or None, optional
        Optional cache for Hamiltonian connections.

    Returns
    -------
    torch.Tensor
        Local energies, shape ``(batch,)``.
    """
    batch = configs.shape[0]

    # --- 1. Evaluate log_amplitude for all input configs (single call) ---
    with torch.no_grad():
        log_amp_x = nqs.log_amplitude(configs)  # (batch,)

    # --- 2. Batch diagonal computation (single vectorised call) ----------
    # diagonal_elements_batch handles device conversion internally,
    # so pass configs directly (avoids unnecessary CPU↔GPU transfer)
    diag_all = hamiltonian.diagonal_elements_batch(configs)
    e_loc = diag_all.to(device).float()  # (batch,)

    # --- 3. Gather all off-diagonal connections --------------------------
    # Move to CPU once for Numba-based get_connections
    configs_cpu = configs.cpu()

    all_connected: list[torch.Tensor] = []
    all_elements: list[torch.Tensor] = []
    # owner_idx[i] stores which original config the i-th connected config
    # belongs to, so we can scatter the results back.
    owner_indices: list[int] = []

    for idx in range(batch):
        config_cpu = configs_cpu[idx]

        if connection_cache is not None:
            connected, elements = connection_cache.get_or_compute(
                config_cpu, hamiltonian
            )
        else:
            connected, elements = hamiltonian.get_connections(config_cpu)

        if connected.numel() > 0:
            all_connected.append(connected)
            all_elements.append(elements)
            owner_indices.extend([idx] * connected.shape[0])

    # --- 4. Single batched NQS evaluation for all connections ------------
    if all_connected:
        all_connected_cat = torch.cat(all_connected, dim=0)  # (N_total, sites)
        all_elements_cat = torch.cat(all_elements, dim=0)  # (N_total,)

        connected_dev = all_connected_cat.to(device).float()
        elements_dev = all_elements_cat.to(device).float()
        owner_dev = torch.tensor(owner_indices, device=device, dtype=torch.long)

        with torch.no_grad():
            log_amp_conn = nqs.log_amplitude(connected_dev)  # (N_total,)

        # psi(x') / psi(x) = exp(log_amp(x') - log_amp(x))
        ratios = torch.exp(log_amp_conn - log_amp_x[owner_dev])
        contributions = elements_dev * ratios  # (N_total,)

        # Scatter-add contributions back to corresponding configs
        e_loc.scatter_add_(0, owner_dev, contributions)

    return e_loc


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def compute_teacher_loss(
    configs: torch.Tensor,
    log_probs_flow: torch.Tensor,
    nqs: nn.Module,
) -> torch.Tensor:
    """Compute the teacher (KL divergence) loss.

    ``L_teacher = -sum_x p_nqs(x) * log p_flow(x)``

    The NQS probabilities are detached (treated as fixed targets).

    Parameters
    ----------
    configs : torch.Tensor
        Sampled configurations, shape ``(batch, num_sites)``.
    log_probs_flow : torch.Tensor
        Flow log-probabilities, shape ``(batch,)``.
    nqs : nn.Module
        Neural quantum state with a ``log_amplitude(x)`` method.

    Returns
    -------
    torch.Tensor
        Scalar teacher loss.
    """
    with torch.no_grad():
        log_amp = nqs.log_amplitude(configs)
        log_prob_nqs = 2.0 * log_amp  # log |psi|^2 (unnormalised)
        # Normalise within the batch
        log_z = torch.logsumexp(log_prob_nqs, dim=0)
        weights = torch.exp(log_prob_nqs - log_z)

    return -(weights * log_probs_flow).sum()


def compute_physics_loss(
    configs: torch.Tensor,
    nqs: nn.Module,
    hamiltonian: Hamiltonian,
    device: torch.device,
    energy_baseline: float,
    baseline_initialized: bool,
    use_energy_baseline: bool,
    ema_decay: float,
    connection_cache: ConnectionCache | None = None,
) -> tuple[torch.Tensor, float, float, bool]:
    """Compute the variational energy (physics) loss.

    ``L_physics = sum_x |psi(x)|^2 * E_loc(x) / Z``

    Uses a running EMA baseline for variance reduction when enabled.

    Parameters
    ----------
    configs : torch.Tensor
        Sampled configurations, shape ``(batch, num_sites)``.
    nqs : nn.Module
        Neural quantum state with a ``log_amplitude(x)`` method.
    hamiltonian : Hamiltonian
        The Hamiltonian operator.
    device : torch.device
        Torch device for computation.
    energy_baseline : float
        Current EMA energy baseline value.
    baseline_initialized : bool
        Whether the baseline has been initialised.
    use_energy_baseline : bool
        Whether to apply variance reduction via EMA baseline.
    ema_decay : float
        Exponential moving average decay for the baseline.
    connection_cache : ConnectionCache or None, optional
        Optional cache for Hamiltonian connections.

    Returns
    -------
    loss : torch.Tensor
        Scalar physics loss.
    mean_energy : float
        Mean local energy (for logging).
    updated_baseline : float
        Updated EMA energy baseline.
    updated_initialized : bool
        Whether the baseline is now initialised.
    """
    e_loc = compute_local_energy(configs, nqs, hamiltonian, device, connection_cache)

    log_amp = nqs.log_amplitude(configs)
    log_prob = 2.0 * log_amp
    log_z = torch.logsumexp(log_prob, dim=0)
    weights = torch.exp(log_prob - log_z)

    mean_energy = float((weights.detach() * e_loc).sum())

    # Variance reduction with EMA baseline
    updated_baseline = energy_baseline
    updated_initialized = baseline_initialized
    if use_energy_baseline:
        if not baseline_initialized:
            updated_baseline = mean_energy
            updated_initialized = True
        else:
            updated_baseline = (
                ema_decay * energy_baseline + (1.0 - ema_decay) * mean_energy
            )
        centred_e = e_loc - updated_baseline
    else:
        centred_e = e_loc

    loss = (weights * centred_e).sum()
    return loss, mean_energy, updated_baseline, updated_initialized


def compute_entropy_loss(
    log_probs_flow: torch.Tensor,
) -> torch.Tensor:
    """Compute the negative entropy of the flow distribution.

    ``L_entropy = sum_x p_flow(x) * log p_flow(x) = -H[p_flow]``

    Minimising this loss maximises the entropy.

    Parameters
    ----------
    log_probs_flow : torch.Tensor
        Flow log-probabilities, shape ``(batch,)``.

    Returns
    -------
    torch.Tensor
        Scalar entropy loss (negative entropy).
    """
    return log_probs_flow.mean()
