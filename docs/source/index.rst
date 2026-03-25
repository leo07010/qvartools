qvartools: Quantum Variational Toolkit
======================================

**qvartools** is a unified Python package for quantum variational methods
applied to molecular ground-state energy estimation. It consolidates
normalizing-flow-guided neural quantum states (NF-NQS), sample-based quantum
diagonalization (SQD), and sample-based Krylov quantum diagonalization (SKQD)
into reusable, well-scoped modules.

Key capabilities:

- Molecular and spin Hamiltonians with efficient matrix-element computation
- Neural quantum state architectures (dense, complex, RBM, transformer)
- Normalizing flows with particle-number conservation
- Physics-guided mixed-objective training
- Sample-based Krylov diagonalization (SKQD) with residual expansion
- Unified solver interface (FCI, CCSD, SQD, SKQD, iterative variants)
- YAML-based experiment configuration with CLI overrides

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/overview
   user_guide/pipelines
   user_guide/yaml_configs

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/hamiltonians
   api/nqs
   api/flows
   api/krylov
   api/diag
   api/solvers
   api/samplers
   api/molecules
   api/pipeline

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/h2_pipeline
   tutorials/beh2_pipeline
   tutorials/custom_solver

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/extending_solvers
   developer_guide/extending_samplers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
