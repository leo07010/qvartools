Normalizing Flows
=================

.. module:: qvartools.flows

The ``flows`` subpackage provides normalizing-flow models for configuration
sampling and physics-guided training.

Flow Networks
-------------

.. autoclass:: qvartools.flows.networks.discrete_flow.DiscreteFlowSampler
   :members:
   :show-inheritance:

.. autoclass:: qvartools.flows.networks.particle_conserving_flow.ParticleConservingFlowSampler
   :members:
   :show-inheritance:

Training
--------

.. autoclass:: qvartools.flows.training.physics_guided_training.PhysicsGuidedFlowTrainer
   :members:
   :show-inheritance:

.. autoclass:: qvartools.flows.training.physics_guided_training.PhysicsGuidedConfig
   :members:

Loss Functions
--------------

.. automodule:: qvartools.flows.training.loss_functions
   :members:

Gumbel Top-k
-------------

.. automodule:: qvartools.flows.training.gumbel_topk
   :members:

Utilities
---------

.. autofunction:: qvartools.flows.networks.particle_conserving_flow.verify_particle_conservation
