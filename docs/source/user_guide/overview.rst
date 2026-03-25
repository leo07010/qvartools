Package Overview
================

qvartools is organized as a collection of loosely coupled subpackages, each
handling one stage of the quantum variational pipeline. This design allows
researchers to use individual components independently or combine them into
end-to-end workflows.

Architecture
------------

The package follows a bottom-up dependency structure:

.. code-block:: text

                        pipeline.py
                       /    |    \    \
                      /     |     \    \
                flows/   krylov/   \    molecules/
               /  |  \     |   \    \       |
              /   |   \    |    \    \      hamiltonians/
         nqs/  hamiltonians/ diag/
              \           |       /
               \          |      /
                \    _utils/    /
                 \      |     /
                  [torch, numpy, scipy]

Key rules:

- **hamiltonians/** depends only on torch, numpy, scipy
- **nqs/** depends only on torch
- **flows/** depends on hamiltonians/ and nqs/
- **krylov/** depends on hamiltonians/ and diag/
- **solvers/** is a leaf consumer -- nothing depends on it
- **pipeline.py** ties everything together at the top

Subpackages
-----------

hamiltonians
^^^^^^^^^^^^

Hamiltonian operators for molecular and spin systems. Provides the abstract
``Hamiltonian`` base class and concrete implementations:

- ``MolecularHamiltonian`` -- second-quantized molecular Hamiltonian via
  Jordan-Wigner mapping with Slater-Condon rules for matrix elements
- ``HeisenbergHamiltonian`` -- anisotropic Heisenberg (XYZ) model
- ``TransverseFieldIsing`` -- transverse-field Ising model
- ``PauliString`` -- individual Pauli operator strings

nqs
^^^

Neural quantum state architectures that parameterize the many-body wavefunction:

- ``DenseNQS`` -- fully connected feedforward network
- ``SignedDenseNQS`` -- amplitude + sign network with explicit sign structure
- ``ComplexNQS`` -- shared feature extractor for amplitude and phase
- ``RBMQuantumState`` -- restricted Boltzmann machine (Carleo & Troyer, 2017)
- ``AutoregressiveTransformer`` -- autoregressive transformer with alpha/beta
  spin channels

flows
^^^^^

Normalizing flows for configuration sampling:

- ``DiscreteFlowSampler`` -- RealNVP flow mapping continuous latent variables
  to discrete binary configurations
- ``ParticleConservingFlowSampler`` -- flow with exact particle-number
  conservation via differentiable top-k (Gumbel-Softmax)
- ``PhysicsGuidedFlowTrainer`` -- joint flow + NQS training with mixed
  objectives (teacher KL, variational energy, entropy regularization)

krylov
^^^^^^

Krylov subspace methods:

- ``SampleBasedKrylovDiagonalization`` -- core SKQD solver constructing Krylov
  states via time evolution
- ``FlowGuidedSKQD`` -- SKQD seeded with normalizing-flow basis
- ``ResidualBasedExpander`` -- iterative basis expansion via residual analysis
- ``SelectedCIExpander`` -- CIPSI-style perturbative basis enrichment

diag
^^^^

Subspace diagonalization utilities:

- ``DiversitySelector`` -- excitation-rank-aware diversity selection
- ``ProjectedHamiltonianBuilder`` -- hash-based projected Hamiltonian
  construction
- ``DavidsonSolver`` -- iterative Davidson eigensolver
- Generalized eigenvalue solvers with GPU acceleration

solvers
^^^^^^^

High-level solver interfaces returning a common ``SolverResult``:

- ``FCISolver`` -- full configuration interaction (exact)
- ``CCSDSolver`` -- coupled cluster singles and doubles (PySCF)
- ``SQDSolver`` -- sample-based quantum diagonalization
- ``SKQDSolver`` -- sample-based Krylov quantum diagonalization
- ``IterativeNFSQDSolver`` -- iterative NF-SQD with eigenvector feedback
- ``IterativeNFSKQDSolver`` -- iterative NF-SKQD with eigenvector feedback

molecules
^^^^^^^^^

Molecular system registry with pre-configured benchmarks:

- 8 molecules from H2 (4 qubits) to C2H4 (28 qubits)
- Automatic integral computation via PySCF
- Factory function ``get_molecule()`` for easy access

Design Principles
-----------------

1. **Immutable configuration.** All config dataclasses are frozen. Adapted
   copies are created via ``dataclasses.replace()``.

2. **Abstract base classes.** ``Hamiltonian``, ``NeuralQuantumState``,
   ``Solver``, and ``Sampler`` define the contracts that concrete
   implementations must satisfy.

3. **Consistent return types.** Solvers return ``SolverResult``, samplers
   return ``SamplerResult``, and the pipeline returns a dictionary with
   documented keys.

4. **No upward dependencies.** Lower-level modules never import from
   higher-level ones.
