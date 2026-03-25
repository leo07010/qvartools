"""
flows --- Normalizing flows for configuration sampling
=======================================================

This subpackage provides normalizing-flow-based samplers for generating
discrete quantum configurations.  It includes a general-purpose discrete
flow, a particle-number-conserving flow for fermionic systems, and a
physics-guided training loop that combines teacher forcing, variational
energy minimisation, and entropy regularisation.

Classes
-------
DiscreteFlowSampler
    RealNVP normalizing flow that maps continuous latent variables to
    discrete binary configurations.
ParticleConservingFlowSampler
    Normalizing flow that exactly conserves alpha and beta particle
    numbers via differentiable top-k selection.
PhysicsGuidedFlowTrainer
    Training orchestrator combining teacher, physics, and entropy
    losses for joint flow + NQS optimisation.
PhysicsGuidedConfig
    Dataclass holding all hyperparameters for physics-guided training.

Functions
---------
verify_particle_conservation
    Validate that sampled configurations satisfy exact particle-number
    constraints.
"""

from qvartools.flows.networks.discrete_flow import DiscreteFlowSampler
from qvartools.flows.networks.particle_conserving_flow import (
    ParticleConservingFlowSampler,
    verify_particle_conservation,
)
from qvartools.flows.training.physics_guided_training import (
    PhysicsGuidedConfig,
    PhysicsGuidedFlowTrainer,
)

__all__ = [
    "DiscreteFlowSampler",
    "ParticleConservingFlowSampler",
    "PhysicsGuidedFlowTrainer",
    "PhysicsGuidedConfig",
    "verify_particle_conservation",
]
