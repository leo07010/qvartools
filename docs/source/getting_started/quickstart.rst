Quickstart
==========

This guide walks through the most common qvartools workflows in a few minutes.

One-Line Pipeline
-----------------

The simplest way to compute a molecular ground-state energy:

.. code-block:: python

   from qvartools import run_molecular_benchmark

   results = run_molecular_benchmark("H2", verbose=True)
   print(f"Energy: {results['final_energy']:.10f} Ha")

This function loads the H2 molecule from the built-in registry, adapts the
pipeline parameters to the system size, trains a normalizing flow, builds a
Krylov subspace, and returns the ground-state energy.

Configurable Pipeline
---------------------

For finer control, instantiate the pipeline directly:

.. code-block:: python

   from qvartools import PipelineConfig, FlowGuidedKrylovPipeline
   from qvartools.molecules import get_molecule

   # Load molecule
   hamiltonian, mol_info = get_molecule("LiH")

   # Configure
   config = PipelineConfig(
       teacher_weight=0.5,
       physics_weight=0.4,
       entropy_weight=0.1,
       max_epochs=200,
   )

   # Run
   pipeline = FlowGuidedKrylovPipeline(
       hamiltonian=hamiltonian,
       config=config,
       auto_adapt=True,
   )
   results = pipeline.run()
   print(f"Energy: {results['final_energy']:.10f} Ha")

Stage-by-Stage Execution
-------------------------

Each pipeline stage can be run independently:

.. code-block:: python

   # Stage 1: Train the NF-NQS model
   history = pipeline.train_flow_nqs(progress=True)

   # Stage 2: Extract and select basis configurations
   basis = pipeline.extract_and_select_basis()

   # Stage 3: Run SKQD Krylov diagonalization
   skqd_results = pipeline.run_subspace_diag(progress=True)

   print(f"Energy: {pipeline.results['final_energy']:.10f} Ha")

Using Individual Solvers
------------------------

Each solver can be used independently:

.. code-block:: python

   from qvartools.molecules import get_molecule
   from qvartools.solvers import FCISolver, SQDSolver, SKQDSolver

   hamiltonian, mol_info = get_molecule("H2")

   # Exact reference
   fci = FCISolver()
   result = fci.solve(hamiltonian, mol_info)
   print(f"FCI energy: {result.energy:.10f} Ha")

   # Sample-based methods
   sqd = SQDSolver(n_samples=2000)
   result = sqd.solve(hamiltonian, mol_info)
   print(f"SQD energy: {result.energy:.10f} Ha")

Spin Hamiltonians
-----------------

qvartools also supports spin-lattice Hamiltonians:

.. code-block:: python

   from qvartools.hamiltonians.spin import HeisenbergHamiltonian

   H = HeisenbergHamiltonian(num_spins=8, Jx=1.0, Jy=1.0, Jz=1.0)
   energy, state = H.exact_ground_state()
   print(f"Heisenberg ground state: {energy:.10f}")

Next Steps
----------

- :doc:`../user_guide/overview` -- understand the package architecture
- :doc:`../user_guide/pipelines` -- learn about the different pipeline methods
- :doc:`../user_guide/yaml_configs` -- configure experiments via YAML files
- :doc:`../tutorials/h2_pipeline` -- detailed H2 tutorial
