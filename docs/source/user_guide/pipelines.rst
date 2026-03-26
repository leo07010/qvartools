Pipeline Methods
================

qvartools provides several pipeline methods that combine NF training, basis
selection, and subspace diagonalization in different configurations. Each
pipeline is available both as a Python API and as a standalone experiment
script.

Pipeline Overview
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Pipeline
     - NF Training
     - Subspace Method
     - Description
   * - NF-SKQD
     - Yes
     - SKQD
     - Full pipeline: train NF, merge with Direct-CI, Krylov expansion
   * - NF-SQD
     - Yes
     - SQD
     - NF training + noise injection + S-CORE batch diagonalization
   * - DCI-SKQD
     - No
     - SKQD
     - Direct-CI configurations + Krylov expansion (no NF)
   * - DCI-SQD
     - No
     - SQD
     - Direct-CI + noise + S-CORE (no NF training)
   * - HI-NQS-SKQD
     - Iterative
     - SKQD
     - Iterative NQS + Krylov with eigenvector feedback
   * - HI-NQS-SQD
     - Iterative
     - SQD
     - Iterative NQS + SQD with eigenvector feedback
   * - NF-Ablation-SKQD
     - Yes
     - SKQD
     - NF-only basis (no Direct-CI merge) — ablation study
   * - NF-Ablation-SQD
     - Yes
     - SQD
     - NF-only basis — ablation study
   * - Trotter-SKQD
     - No
     - SKQD
     - HF-only reference state — baseline/ablation

The FlowGuidedKrylovPipeline
-----------------------------

The main pipeline class orchestrates four stages:

**Stage 1: Train** — Joint physics-guided training of the normalizing flow and
NQS using a mixed objective (teacher KL-divergence + variational energy +
entropy regularization).

**Stage 2: Select** — Extract accumulated basis configurations from the trained
flow and apply diversity-aware selection to ensure representation across
excitation ranks.

**Stage 3: Expand** — Optionally enlarge the basis via residual analysis or
CIPSI-style perturbative selection.

**Stage 4: Refine** — Run either SKQD (Krylov subspace) or SQD (batch
diagonalization) to compute the ground-state energy.

.. code-block:: python

   from qvartools import PipelineConfig, FlowGuidedKrylovPipeline
   from qvartools.molecules import get_molecule

   hamiltonian, mol_info = get_molecule("BeH2")

   config = PipelineConfig(
       skip_nf_training=False,
       subspace_mode="classical_krylov",   # "classical_krylov", "skqd", or "sqd"
       teacher_weight=0.5,
       physics_weight=0.4,
       entropy_weight=0.1,
   )

   pipeline = FlowGuidedKrylovPipeline(
       hamiltonian=hamiltonian,
       config=config,
       auto_adapt=True,  # auto-scale parameters to system size
   )

   results = pipeline.run()

Iterative Pipelines
--------------------

The HI-NQS pipelines use an iterative loop where the ground-state eigenvector
from each diagonalization is fed back as a training target for the next NQS
iteration:

.. code-block:: python

   from qvartools.solvers import IterativeNFSQDSolver
   from qvartools.molecules import get_molecule

   hamiltonian, mol_info = get_molecule("H2")

   solver = IterativeNFSQDSolver(
       n_iterations=30,
       n_samples=5000,
       convergence_tol=1e-6,
   )

   result = solver.solve(hamiltonian, mol_info)
   print(f"Energy: {result.energy:.10f} Ha")
   print(f"Converged: {result.converged}")

Running Experiment Scripts
--------------------------

Each pipeline has a corresponding experiment script in
``experiments/methods/``:

.. code-block:: bash

   # Run flow+CI -> Krylov on H2 with default parameters
   python experiments/methods/flow_ci_krylov.py h2

   # Run with a YAML config file
   python experiments/methods/flow_ci_krylov.py --config experiments/configs/flow_ci_krylov.yaml

   # Override parameters via CLI
   python experiments/methods/flow_ci_krylov.py lih --max-epochs 200 --teacher-weight 0.6

See :doc:`yaml_configs` for details on the configuration system.
