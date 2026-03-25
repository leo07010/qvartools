# Architecture Overview

## Design Philosophy

qvartools follows a **bottom-up, fine-grained module** design:

1. **Many small files over few large files.** Each module has a single responsibility and typically stays under 400 lines.
2. **Immutable configuration.** All config dataclasses are frozen. Adapted copies are created via `dataclasses.replace`, never mutation.
3. **Abstract base classes define contracts.** `Hamiltonian`, `NeuralQuantumState`, `Solver`, and `Sampler` are ABCs that concrete implementations extend.
4. **No upward dependencies.** Lower-level subpackages never import from higher-level ones. `hamiltonians` and `nqs` depend on nothing within qvartools; `pipeline` sits at the top and orchestrates everything.
5. **Consistent return types.** Solvers return `SolverResult`, samplers return `SamplerResult`, and the pipeline returns a plain dictionary with documented keys.

## Module Dependency Graph

```
                        pipeline.py
                       /    |    \    \
                      /     |     \    \
                flows/   krylov/   \    molecules/
               /  |  \     |   \    \       |
              /   |   \    |    \    \      hamiltonians/
         nqs/  hamiltonians/ postprocessing/
              \           |       /
               \          |      /
                \    _utils/    /
                 \      |     /
                  \     |    /
                   [torch, numpy, scipy]

    solvers/ -----> flows/, krylov/, postprocessing/, hamiltonians/, molecules/
    samplers/ ----> flows/, hamiltonians/
```

Key dependency rules:

- `hamiltonians/` depends only on torch, numpy, scipy (and optionally numba)
- `nqs/` depends only on torch
- `flows/` depends on `hamiltonians/` and `nqs/`
- `krylov/` depends on `hamiltonians/` and `postprocessing/`
- `postprocessing/` depends on `hamiltonians/` (for projected Hamiltonian construction)
- `solvers/` depends on most subpackages but is a leaf consumer -- nothing depends on it
- `pipeline.py` orchestrates `flows/`, `krylov/`, `postprocessing/`, `molecules/`, and `nqs/`
- `_utils/` is internal and used by several subpackages

## Data Flow Through the Pipeline

The `FlowGuidedKrylovPipeline` executes four stages sequentially:

```
Stage 1: TRAIN              Stage 2: SELECT           Stage 3: EXPAND          Stage 4: REFINE
------------------          -----------------         -----------------        -----------------
PhysicsGuidedFlow   --->    DiversitySelector  --->   ResidualExpander   --->  FlowGuidedSKQD
Trainer                                               or SelectedCI
                                                      Expander
Input:                      Input:                    Input:                   Input:
  Hamiltonian                 Accumulated configs       Selected configs         Expanded configs
  Flow + NQS models           Reference state           Hamiltonian              Hamiltonian

Output:                     Output:                   Output:                  Output:
  Trained flow + NQS          Diverse subset            Enriched basis           Ground-state
  Accumulated configs         (excitation-rank          (residual or PT2         energy +
  Training history            bucketed, Hamming         selected new             eigenvalues
                              filtered)                 configs)
```

### Stage 1: Physics-Guided NF-NQS Training

The `PhysicsGuidedFlowTrainer` jointly trains a normalizing flow and a neural quantum state using a weighted combination of three losses:

- **Teacher loss**: trains the flow to match the NQS distribution (KL divergence)
- **Physics loss**: minimizes the variational energy via local energy estimation
- **Entropy loss**: encourages exploration by maximizing flow entropy

For molecular systems, the `ParticleConservingFlowSampler` guarantees that all generated configurations have the correct number of alpha and beta electrons. Temperature annealing gradually sharpens the sampling distribution.

### Stage 2: Diversity-Aware Basis Selection

The `DiversitySelector` takes the accumulated configuration pool from training and selects a compact, representative subset using:

- **Excitation-rank bucketing**: allocates budget fractions to singles, doubles, triples, and higher excitations relative to the Hartree-Fock reference
- **Hamming-distance filtering**: within each bucket, greedily selects configurations that are maximally diverse

### Stage 3: Residual / Perturbative Basis Expansion

Two expansion strategies are available:

- **ResidualBasedExpander**: computes the residual vector r = (H - E)|psi> and adds configurations with the largest residual components
- **SelectedCIExpander**: uses second-order perturbative (CIPSI-style) importance to rank candidate configurations

Both iterate until the energy improvement per iteration falls below a threshold.

### Stage 4: SKQD Refinement

The `FlowGuidedSKQD` solver seeds the Krylov subspace with the NF-expanded basis and iteratively constructs Krylov vectors via time evolution e^{-iHt}|psi>. At each step it samples configurations from the evolved state, builds the projected Hamiltonian and overlap matrices, regularizes the overlap, and solves the generalized eigenvalue problem.

## Key Abstractions

### Hamiltonian

The `Hamiltonian` ABC requires two methods:

- `diagonal_element(config) -> Tensor`: the diagonal matrix element for a configuration
- `get_connections(config) -> (configs, elements)`: all off-diagonal connected states and their matrix elements

Concrete implementations: `MolecularHamiltonian` (Jordan-Wigner), `HeisenbergHamiltonian` (XXZ/XYZ), `TransverseFieldIsing`.

### NeuralQuantumState

The `NeuralQuantumState` ABC (extends `nn.Module`) requires:

- `log_amplitude(x) -> Tensor`: log of the wavefunction amplitude
- `phase(x) -> Tensor`: wavefunction phase in radians

The base class provides `log_psi`, `psi`, `probability`, and `normalized_probability`. Concrete implementations: `DenseNQS`, `SignedDenseNQS`, `ComplexNQS`, `RBMQuantumState`, `AutoregressiveTransformer`.

### Solver

The `Solver` ABC requires a single method:

- `solve(hamiltonian, mol_info) -> SolverResult`

All solvers return a frozen `SolverResult` dataclass with `energy`, `diag_dim`, `wall_time`, `method`, `converged`, and `metadata`. Concrete implementations: `FCISolver`, `CCSDSolver`, `SQDSolver`, `SKQDSolver`, `IterativeNFSQDSolver`, `IterativeNFSKQDSolver`.

### Sampler

The `Sampler` ABC requires:

- `sample(n_samples) -> SamplerResult`

Returns a `SamplerResult` with `configs`, `counts`, and `metadata`. Concrete implementations: `NFSampler`, `TrotterSampler`.

## Extension Points

### Adding a New Hamiltonian

1. Create a new file in `src/qvartools/hamiltonians/`
2. Subclass `Hamiltonian` and implement `diagonal_element` and `get_connections`
3. Export from `hamiltonians/__init__.py`

```python
from qvartools.hamiltonians.base import Hamiltonian

class MyHamiltonian(Hamiltonian):
    def __init__(self, num_sites, ...):
        super().__init__(num_sites=num_sites, local_dim=2)

    def diagonal_element(self, config):
        ...  # Return scalar tensor

    def get_connections(self, config):
        ...  # Return (connected_configs, matrix_elements)
```

### Adding a New NQS Architecture

1. Create a new file in `src/qvartools/nqs/`
2. Subclass `NeuralQuantumState` and implement `log_amplitude` and `phase`
3. Export from `nqs/__init__.py`

### Adding a New Solver

1. Create a new file in `src/qvartools/solvers/`
2. Subclass `Solver` and implement `solve(hamiltonian, mol_info) -> SolverResult`
3. Export from `solvers/__init__.py`

### Adding a New Molecule

1. Define the geometry in `src/qvartools/molecules/registry.py`
2. Write a factory function `_make_<name>` that calls `compute_molecular_integrals`
3. Add the entry to `MOLECULE_REGISTRY`
