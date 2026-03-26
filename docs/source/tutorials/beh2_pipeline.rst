Tutorial: BeH2 Ground-State Energy
===================================

This tutorial demonstrates the pipeline on BeH2 (beryllium dihydride), a
14-qubit system that is more challenging than H2 and shows the value of
the NF-guided approach.

BeH2 has a larger Hilbert space (:math:`\binom{7}{1}^2 \times \binom{7}{2}^2
= 2{,}025` valid configurations in the spin-orbital basis), which makes
the normalizing flow's ability to focus sampling on important configurations
more valuable.

Setup and Configuration
-----------------------

.. code-block:: python

   from qvartools import PipelineConfig, FlowGuidedKrylovPipeline
   from qvartools.molecules import get_molecule
   from qvartools.solvers import FCISolver

   # Load BeH2
   hamiltonian, mol_info = get_molecule("BeH2")
   print(f"Qubits: {mol_info['n_qubits']}")

   # Exact reference
   fci_result = FCISolver().solve(hamiltonian, mol_info)
   print(f"Exact energy: {fci_result.energy:.10f} Ha")

Running the Full Pipeline
--------------------------

.. code-block:: python

   config = PipelineConfig(
       teacher_weight=0.5,
       physics_weight=0.4,
       entropy_weight=0.1,
       max_epochs=200,
       samples_per_batch=1500,
   )

   pipeline = FlowGuidedKrylovPipeline(
       hamiltonian=hamiltonian,
       config=config,
       exact_energy=fci_result.energy,
       auto_adapt=True,
   )

   results = pipeline.run()

   error_mha = (results["final_energy"] - fci_result.energy) * 1000.0
   print(f"Energy: {results['final_energy']:.10f} Ha")
   print(f"Error: {error_mha:.4f} mHa")

Comparing Methods
-----------------

Try running different pipeline variants on BeH2 to compare:

.. code-block:: python

   # DCI-SKQD (no NF training)
   dci_config = PipelineConfig(
       skip_nf_training=True,
       subspace_mode="classical_krylov",
   )
   dci_pipeline = FlowGuidedKrylovPipeline(
       hamiltonian=hamiltonian,
       config=dci_config,
       exact_energy=fci_result.energy,
       auto_adapt=True,
   )
   dci_results = dci_pipeline.run()

The NF-SKQD pipeline typically achieves lower error than DCI-SKQD because the
normalizing flow discovers important high-excitation configurations that
deterministic CI (HF + singles + doubles) misses.

Using YAML Configs
------------------

.. code-block:: bash

   # Run from the command line
   python experiments/methods/flow_ci_krylov.py beh2 --config experiments/configs/flow_ci_krylov.yaml
   python experiments/methods/direct_ci_krylov.py beh2 --config experiments/configs/direct_ci_krylov.yaml

Interpreting Results
--------------------

Key metrics to examine:

- **Error in mHa**: values below 1.6 mHa indicate chemical accuracy
- **Basis size**: the number of configurations used in the final diagonalization
- **SKQD convergence**: the energy at each Krylov step should decrease monotonically
- **Wall time**: NF training adds overhead but typically reduces the required basis size
