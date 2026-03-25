Neural Quantum States
=====================

.. module:: qvartools.nqs

The ``nqs`` subpackage provides neural quantum state architectures that
parameterize the many-body wavefunction as a neural network.

Base Class
----------

.. autoclass:: qvartools.nqs.neural_state.NeuralQuantumState
   :members:
   :undoc-members:
   :show-inheritance:

Dense Architectures
-------------------

.. autoclass:: qvartools.nqs.architectures.dense.DenseNQS
   :members:
   :show-inheritance:

.. autoclass:: qvartools.nqs.architectures.dense.SignedDenseNQS
   :members:
   :show-inheritance:

.. autoclass:: qvartools.nqs.architectures.complex_nqs.ComplexNQS
   :members:
   :show-inheritance:

Restricted Boltzmann Machine
-----------------------------

.. autoclass:: qvartools.nqs.architectures.rbm.RBMQuantumState
   :members:
   :show-inheritance:

Autoregressive Transformer
--------------------------

.. autoclass:: qvartools.nqs.transformer.autoregressive.AutoregressiveTransformer
   :members:
   :show-inheritance:

Utilities
---------

.. autofunction:: qvartools.nqs.compile_nqs
