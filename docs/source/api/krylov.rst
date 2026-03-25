Krylov Methods
==============

.. module:: qvartools.krylov

The ``krylov`` subpackage implements sample-based Krylov quantum
diagonalization (SKQD) and basis expansion methods.

Core SKQD
----------

.. autoclass:: qvartools.krylov.basis.skqd.SampleBasedKrylovDiagonalization
   :members:
   :show-inheritance:

.. autoclass:: qvartools.krylov.basis.skqd.SKQDConfig
   :members:

Flow-Guided SKQD
-----------------

.. autoclass:: qvartools.krylov.basis.flow_guided.FlowGuidedSKQD
   :members:
   :show-inheritance:

Basis Expansion
---------------

.. autoclass:: qvartools.krylov.expansion.residual_config.ResidualExpansionConfig
   :members:

.. autoclass:: qvartools.krylov.expansion.residual_expander.ResidualBasedExpander
   :members:
   :show-inheritance:

.. autoclass:: qvartools.krylov.expansion.selected_ci_expander.SelectedCIExpander
   :members:
   :show-inheritance:

Basis Sampling
--------------

.. autoclass:: qvartools.krylov.basis.sampler.KrylovBasisSampler
   :members:
   :show-inheritance:

Krylov Expansion Utilities
--------------------------

.. automodule:: qvartools.krylov.expansion.krylov_expand
   :members:
