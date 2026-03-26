Tutorial: H2 Ground-State Energy
================================

This tutorial walks through computing the ground-state energy of H2 using the
full NF-SKQD pipeline. H2 is the simplest molecule in the registry (4 qubits),
making it ideal for learning the workflow.

Setup
-----

.. code-block:: python

   from qvartools import PipelineConfig, FlowGuidedKrylovPipeline
   from qvartools.molecules import get_molecule
   from qvartools.solvers import FCISolver

Step 1: Load the Molecule
--------------------------

.. code-block:: python

   hamiltonian, mol_info = get_molecule("H2")

   print(f"Molecule: {mol_info['name']}")
   print(f"Qubits: {mol_info['n_qubits']}")
   print(f"Basis set: {mol_info['basis']}")

This returns a ``MolecularHamiltonian`` built from PySCF-computed integrals
and a metadata dictionary.

Step 2: Compute the Exact Reference
------------------------------------

.. code-block:: python

   fci_result = FCISolver().solve(hamiltonian, mol_info)
   exact_energy = fci_result.energy
   print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")

For H2 in sto-3g, the exact ground-state energy is approximately -1.137 Ha.

Step 3: Configure the Pipeline
-------------------------------

.. code-block:: python

   config = PipelineConfig(
       skip_nf_training=False,
       subspace_mode="classical_krylov",
       teacher_weight=0.5,
       physics_weight=0.4,
       entropy_weight=0.1,
   )

   pipeline = FlowGuidedKrylovPipeline(
       hamiltonian=hamiltonian,
       config=config,
       exact_energy=exact_energy,
       auto_adapt=True,    # auto-scale parameters to H2's size
   )

With ``auto_adapt=True``, the pipeline automatically scales network sizes,
training epochs, and sampling budgets to the system's Hilbert-space dimension.

Step 4: Train the NF-NQS Model
-------------------------------

.. code-block:: python

   history = pipeline.train_flow_nqs(progress=True)

   print(f"Epochs: {len(history.get('total_loss', []))}")
   print(f"Final loss: {history['total_loss'][-1]:.4f}")

The training loop optimizes a joint objective combining:

- **Teacher loss**: KL divergence between the flow and the NQS
- **Physics loss**: variational energy estimate
- **Entropy loss**: encourages exploration of configuration space

Step 5: Extract and Select Basis
---------------------------------

.. code-block:: python

   basis = pipeline.extract_and_select_basis()
   print(f"Selected {basis.shape[0]} configurations")

The diversity selector ensures representation across excitation ranks
(singles, doubles, and higher excitations relative to the Hartree-Fock state).

Step 6: SKQD Diagonalization
-----------------------------

.. code-block:: python

   skqd_results = pipeline.run_subspace_diag(progress=True)

   final_energy = pipeline.results["final_energy"]
   error_mha = (final_energy - exact_energy) * 1000.0
   print(f"Final energy: {final_energy:.10f} Ha")
   print(f"Error: {error_mha:.4f} mHa")

The SKQD solver constructs Krylov states via time evolution
:math:`|\\psi_k\\rangle = e^{-iH \\Delta t \\cdot k} |\\psi_0\\rangle`,
samples configurations from each state, builds the projected Hamiltonian
in the combined basis, and solves the generalized eigenvalue problem.

For H2, the error should be well within chemical accuracy (1.6 mHa).

Running from the Command Line
------------------------------

The same pipeline can be run as a standalone script:

.. code-block:: bash

   python experiments/methods/flow_ci_krylov.py h2

   # Or with a YAML config
   python experiments/methods/flow_ci_krylov.py --config experiments/configs/flow_ci_krylov.yaml
