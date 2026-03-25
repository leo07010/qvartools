# API Reference

Complete reference for every public module in qvartools, organized by subpackage.

---

## Top-Level (`qvartools`)

### `PipelineConfig`

Frozen dataclass holding all hyperparameters for the flow-guided Krylov pipeline.

```python
@dataclass(frozen=True)
class PipelineConfig:
    use_particle_conserving_flow: bool = True
    nf_hidden_dims: List[int] = [128, 64]
    nqs_hidden_dims: List[int] = [128, 64]
    samples_per_batch: int = 500
    num_batches: int = 10
    max_epochs: int = 200
    min_epochs: int = 50
    convergence_threshold: float = 0.01
    teacher_weight: float = 1.0
    physics_weight: float = 0.0
    entropy_weight: float = 0.0
    flow_lr: float = 1e-3
    nqs_lr: float = 1e-3
    max_accumulated_basis: int = 5000
    use_diversity_selection: bool = True
    max_diverse_configs: int = 1000
    rank_2_fraction: float = 0.40
    use_residual_expansion: bool = True
    residual_iterations: int = 10
    residual_configs_per_iter: int = 100
    residual_threshold: float = 1e-4
    use_perturbative_selection: bool = True
    max_krylov_dim: int = 10
    time_step: float = 0.1
    shots_per_krylov: int = 1000
    skqd_regularization: float = 1e-8
    skip_skqd: bool = False
    device: str = "cpu"
```

**Methods:**

- `adapt_to_system_size(n_valid_configs: int) -> PipelineConfig` -- Return a new config with parameters scaled for the given Hilbert-space size. Classifies into four tiers: small (< 1000), medium (< 100000), large (< 1000000), very_large.

### `FlowGuidedKrylovPipeline`

Main orchestrator for the four-stage flow-guided Krylov diagonalization pipeline.

```python
FlowGuidedKrylovPipeline(
    hamiltonian: Hamiltonian,
    config: PipelineConfig,
    exact_energy: float | None = None,
    auto_adapt: bool = True,
)
```

**Methods:**

- `train_flow_nqs(progress: bool = True) -> dict` -- Stage 1: Joint physics-guided training. Returns training history dict.
- `extract_and_select_basis() -> torch.Tensor` -- Stage 2: Extract accumulated basis and apply diversity selection.
- `run_residual_expansion(basis: torch.Tensor) -> torch.Tensor` -- Stage 3: Expand basis via residual or perturbative selection.
- `run_skqd(basis: torch.Tensor, progress: bool = True) -> dict` -- Stage 4: Krylov refinement. Returns dict with `energy`, `eigenvalues`, `basis_size`, `krylov_dim`, `energies_per_step`.
- `run(progress: bool = True) -> dict` -- Execute the complete four-stage pipeline. Returns dict with `training_history`, `basis_after_selection`, `basis_after_expansion`, `skqd_results`, `final_energy`, `exact_energy`, `error_mha`, `config`.

### `run_molecular_benchmark`

```python
run_molecular_benchmark(
    molecule: str,
    config: PipelineConfig | None = None,
    verbose: bool = True,
) -> dict
```

Convenience function: loads a molecule from the registry and runs the full pipeline. Raises `KeyError` if the molecule is not found.

---

## `qvartools.hamiltonians`

### `Hamiltonian` (ABC)

```python
Hamiltonian(num_sites: int, local_dim: int = 2)
```

**Attributes:** `num_sites`, `local_dim`, `hilbert_dim`

**Abstract methods:**

- `diagonal_element(config: Tensor) -> Tensor` -- Diagonal matrix element.
- `get_connections(config: Tensor) -> tuple[Tensor, Tensor]` -- Off-diagonal connected states and matrix elements.

**Concrete methods:**

- `matrix_element(config_i: Tensor, config_j: Tensor) -> Tensor` -- Single matrix element.
- `matrix_elements(configs_bra: Tensor, configs_ket: Tensor) -> Tensor` -- Full matrix block.
- `to_dense(device: str = "cpu") -> Tensor` -- Full dense Hamiltonian matrix.
- `to_sparse(device: str = "cpu") -> scipy.sparse.csr_matrix` -- Sparse CSR matrix.
- `exact_ground_state(device: str = "cpu") -> tuple[float, Tensor]` -- Exact ground state via dense diagonalization.
- `ground_state_sparse(k: int = 1, device: str = "cpu") -> tuple[ndarray, ndarray]` -- Lowest k eigenstates via sparse diagonalization.

### `PauliString`

```python
PauliString(paulis: List[str], coefficient: complex = 1.0)
```

**Attributes:** `paulis`, `coefficient`, `num_qubits`

**Methods:**

- `apply(config: Tensor) -> tuple[Tensor | None, complex]` -- Apply the Pauli string to a computational-basis state.
- `is_diagonal() -> bool` -- True if the string contains only I and Z.

### `MolecularIntegrals`

Frozen dataclass containing one- and two-electron integrals.

```python
@dataclass(frozen=True)
class MolecularIntegrals:
    h1e: np.ndarray           # (n_orb, n_orb) one-electron integrals
    h2e: np.ndarray           # (n_orb, n_orb, n_orb, n_orb) two-electron integrals
    nuclear_repulsion: float  # Nuclear repulsion energy
    n_orbitals: int           # Number of spatial orbitals
    n_alpha: int              # Number of alpha electrons
    n_beta: int               # Number of beta electrons
    n_electrons: int          # Total electron count
```

### `MolecularHamiltonian`

```python
MolecularHamiltonian(integrals: MolecularIntegrals, device: str = "cpu")
```

Second-quantized molecular Hamiltonian via Jordan-Wigner mapping. Extends `Hamiltonian`. Has `num_sites = 2 * n_orbitals` (spin-orbitals).

**Additional attributes:** `integrals`

### `compute_molecular_integrals`

```python
compute_molecular_integrals(
    geometry: list[tuple[str, tuple[float, float, float]]],
    basis: str,
    charge: int = 0,
    spin: int = 0,
) -> MolecularIntegrals
```

Run RHF via PySCF and return molecular integrals. Requires PySCF.

### `HeisenbergHamiltonian`

```python
HeisenbergHamiltonian(
    num_spins: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    h_x: float | array = 0.0,
    h_y: float | array = 0.0,
    h_z: float | array = 0.0,
    periodic: bool = True,
)
```

Anisotropic Heisenberg (XYZ) model on a 1-D chain.

**Additional methods:**

- `diagonal_elements_batch(configs: Tensor) -> Tensor` -- Vectorized diagonal elements for a batch.

### `TransverseFieldIsing`

```python
TransverseFieldIsing(
    num_spins: int,
    V: float = 1.0,
    h: float = 1.0,
    L: int = 1,
    periodic: bool = True,
)
```

Transverse-field Ising model. `V` is the ZZ interaction strength, `h` the transverse field, `L` the interaction range.

---

## `qvartools.nqs`

### `NeuralQuantumState` (ABC)

```python
NeuralQuantumState(num_sites: int, local_dim: int = 2, complex_output: bool = False)
```

Extends `torch.nn.Module`.

**Abstract methods:**

- `log_amplitude(x: Tensor) -> Tensor` -- Log-amplitude for batch of configs, shape `(batch,)`.
- `phase(x: Tensor) -> Tensor` -- Phase in radians, shape `(batch,)`.

**Concrete methods:**

- `log_psi(x: Tensor) -> Tensor | tuple[Tensor, Tensor]` -- Log-wavefunction (real) or (log_amp, phase) tuple (complex).
- `psi(x: Tensor) -> Tensor` -- Full wavefunction values.
- `probability(x: Tensor) -> Tensor` -- Unnormalized Born-rule probability |psi|^2.
- `normalized_probability(x: Tensor, basis_set: Tensor) -> Tensor` -- Normalized probabilities over a basis set.
- `encode_configuration(config: Tensor) -> Tensor` -- Convert config to float32 input.
- `forward(x: Tensor)` -- Delegates to `log_psi`.

### `DenseNQS`

```python
DenseNQS(num_sites: int, hidden_dims: List[int] | None = None, complex_output: bool = False)
```

Fully connected feedforward NQS. Default `hidden_dims = [128, 64]`. Amplitude network uses Tanh output scaled by a learnable `log_amp_scale` parameter.

### `SignedDenseNQS`

```python
SignedDenseNQS(num_sites: int, hidden_dims: List[int] | None = None)
```

Dense NQS with explicit sign structure. Shared feature extractor feeds amplitude head (Softplus) and sign head (sigmoid-thresholded). Always `complex_output=True`.

### `ComplexNQS`

```python
ComplexNQS(num_sites: int, hidden_dims: List[int] | None = None)
```

Dense NQS with shared feature extractor for amplitude and phase.

### `RBMQuantumState`

```python
RBMQuantumState(num_sites: int, n_hidden: int = 128)
```

Restricted Boltzmann Machine NQS (Carleo & Troyer, 2017).

### `AutoregressiveTransformer`

```python
AutoregressiveTransformer(
    num_sites: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
)
```

Autoregressive transformer NQS with alpha/beta spin channels.

### `compile_nqs`

```python
compile_nqs(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module
```

Apply `torch.compile` with graceful fallback on unsupported platforms.

---

## `qvartools.flows`

### `DiscreteFlowSampler`

```python
DiscreteFlowSampler(num_sites: int, hidden_dims: List[int] | None = None)
```

RealNVP normalizing flow mapping continuous latent variables to discrete binary configurations via thresholding. Uses a multi-modal Gaussian prior.

**Key methods:**

- `sample(batch_size: int) -> tuple[Tensor, Tensor]` -- Returns `(all_configs, unique_configs)`.
- `log_prob(x: Tensor) -> Tensor` -- Log-probability under the flow.

### `ParticleConservingFlowSampler`

```python
ParticleConservingFlowSampler(
    num_sites: int,
    n_alpha: int,
    n_beta: int,
    hidden_dims: List[int] | None = None,
)
```

Normalizing flow that exactly conserves alpha and beta particle numbers via differentiable top-k selection (Gumbel-Softmax). All sampled configurations are guaranteed valid Slater determinants.

**Key methods:**

- `sample(batch_size: int) -> tuple[Tensor, Tensor]` -- Returns `(all_configs, unique_configs)`.
- `log_prob(x: Tensor) -> Tensor` -- Log-probability under the flow.
- `set_temperature(temp: float) -> None` -- Adjust Gumbel-Softmax temperature.

### `PhysicsGuidedFlowTrainer`

```python
PhysicsGuidedFlowTrainer(
    flow: nn.Module,
    nqs: NeuralQuantumState,
    hamiltonian: Hamiltonian,
    config: PhysicsGuidedConfig,
    device: str = "cpu",
)
```

Training orchestrator for joint flow + NQS optimization.

**Key methods:**

- `train(progress: bool = True) -> dict` -- Run the full training loop. Returns history dict with per-epoch metrics.

**Key attributes:**

- `accumulated_basis: Tensor | None` -- Unique configurations accumulated during training.

### `PhysicsGuidedConfig`

Frozen dataclass for `PhysicsGuidedFlowTrainer` hyperparameters.

```python
@dataclass(frozen=True)
class PhysicsGuidedConfig:
    samples_per_batch: int = 500
    num_batches: int = 10
    num_epochs: int = 200
    min_epochs: int = 50
    convergence_threshold: float = 0.01
    flow_lr: float = 1e-3
    nqs_lr: float = 1e-3
    teacher_weight: float = 1.0
    physics_weight: float = 0.0
    entropy_weight: float = 0.0
    use_energy_baseline: bool = True
    device: str = "cpu"
```

### `verify_particle_conservation`

```python
verify_particle_conservation(
    configs: Tensor,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
) -> tuple[bool, dict]
```

Validate that sampled configurations satisfy exact particle-number constraints. Returns `(all_valid, stats_dict)`.

---

## `qvartools.krylov`

### `SKQDConfig`

Frozen dataclass for SKQD hyperparameters.

```python
@dataclass(frozen=True)
class SKQDConfig:
    max_krylov_dim: int = 10
    time_step: float = 0.1
    total_evolution_time: float = 1.0
    shots_per_krylov: int = 1000
    use_cumulative_basis: bool = True
    num_eigenvalues: int = 1
    which_eigenvalues: str = "SA"
    regularization: float = 1e-8
```

### `SampleBasedKrylovDiagonalization`

```python
SampleBasedKrylovDiagonalization(
    hamiltonian: Hamiltonian,
    config: SKQDConfig,
    initial_state: np.ndarray | None = None,
)
```

Core SKQD solver. Constructs Krylov states via time evolution, samples configurations, and solves the projected eigenvalue problem.

**Methods:**

- `run() -> tuple[ndarray, dict]` -- Run SKQD. Returns `(eigenvalues, info_dict)`.

### `FlowGuidedSKQD`

```python
FlowGuidedSKQD(
    hamiltonian: Hamiltonian,
    config: SKQDConfig,
    nf_basis: Tensor,
)
```

SKQD variant seeded with normalizing-flow basis states.

**Methods:**

- `run_with_nf(progress: bool = True) -> dict` -- Run SKQD with NF seeding. Returns dict with `energy`, `eigenvalues`, `basis_size`, `krylov_dim`, `energies_per_step`, `nf_energy`, `basis_configs`.

### `ResidualExpansionConfig`

```python
@dataclass(frozen=True)
class ResidualExpansionConfig:
    max_configs_per_iter: int = 100
    residual_threshold: float = 1e-4
    max_iterations: int = 10
    max_basis_size: int = 5000
    min_energy_improvement_mha: float = 0.01
    stagnation_patience: int = 3
```

### `ResidualBasedExpander`

```python
ResidualBasedExpander(hamiltonian: Hamiltonian, config: ResidualExpansionConfig)
```

**Methods:**

- `expand_basis(current_basis: Tensor, energy: float, eigenvector: ndarray) -> tuple[Tensor, dict]` -- Iteratively expand basis via residual analysis. Returns `(expanded_basis, stats)`.

### `SelectedCIExpander`

```python
SelectedCIExpander(hamiltonian: Hamiltonian, config: ResidualExpansionConfig)
```

CIPSI-style selected-CI basis expansion using second-order perturbative importance.

**Methods:**

- `expand_basis(current_basis: Tensor, energy: float, eigenvector: ndarray) -> tuple[Tensor, dict]` -- Same interface as `ResidualBasedExpander`.

### `KrylovBasisSampler`

```python
KrylovBasisSampler(
    hamiltonian: Hamiltonian,
    num_qubits: int,
    shots: int = 1000,
    time_step: float = 0.1,
)
```

Samples configurations from Krylov-evolved states `e^{-iH*dt*k}|psi_0>`.

---

## `qvartools.postprocessing`

### `DiversityConfig`

```python
@dataclass(frozen=True)
class DiversityConfig:
    max_configs: int = 1000
    rank_2_fraction: float = 0.40
    rank_4_plus_fraction: float = 0.15
```

### `DiversitySelector`

```python
DiversitySelector(
    config: DiversityConfig,
    reference: Tensor,
    n_orbitals: int,
)
```

**Methods:**

- `select(configs: Tensor) -> tuple[Tensor, dict]` -- Select a diverse subset. Returns `(selected_configs, selection_stats)`.

### `solve_generalized_eigenvalue`

```python
solve_generalized_eigenvalue(
    H: ndarray | sparse,
    S: ndarray | sparse,
    k: int = 1,
    which: str = "SA",
    use_gpu: bool = False,
) -> tuple[ndarray, ndarray]
```

Solve the generalized eigenvalue problem Hv = ESv. Returns `(eigenvalues, eigenvectors)`.

### `compute_ground_state_energy`

```python
compute_ground_state_energy(H: ndarray | sparse) -> float
```

Extract the ground-state energy from a Hamiltonian matrix.

### `regularize_overlap_matrix`

```python
regularize_overlap_matrix(S: ndarray, epsilon: float = 1e-8) -> ndarray
```

Tikhonov regularization to ensure positive-definiteness.

### `DavidsonSolver`

Iterative Davidson eigensolver for large sparse Hermitian matrices.

### `ProjectedHamiltonianConfig`

```python
@dataclass(frozen=True)
class ProjectedHamiltonianConfig:
    use_sparse: bool = True
    batch_size: int = 1000
```

### `ProjectedHamiltonianBuilder`

```python
ProjectedHamiltonianBuilder(
    hamiltonian: Hamiltonian,
    config: ProjectedHamiltonianConfig | None = None,
)
```

Builds the projected Hamiltonian matrix H_ij = <x_i|H|x_j> in a sampled basis using hash-based O(1) lookup.

---

## `qvartools.solvers`

### `SolverResult`

```python
@dataclass(frozen=True)
class SolverResult:
    energy: float        # Ground-state energy (Hartree)
    diag_dim: int        # Diagonalization subspace dimension
    wall_time: float     # Wall-clock time (seconds)
    method: str          # Solver method name
    converged: bool      # Whether solver converged
    metadata: dict = {}  # Method-specific information
```

### `Solver` (ABC)

```python
class Solver(ABC):
    @abstractmethod
    def solve(self, hamiltonian: Hamiltonian, mol_info: dict) -> SolverResult: ...
```

### `FCISolver`

```python
FCISolver(max_configs: int = 500_000)
```

Full configuration interaction. Uses PySCF when available; falls back to dense diagonalization.

### `CCSDSolver`

```python
CCSDSolver()
```

Coupled cluster singles and doubles via PySCF. Requires PySCF.

### `SQDSolver`

```python
SQDSolver(
    n_samples: int = 5000,
    training_config: dict | None = None,
    device: str = "cpu",
)
```

Sample-based quantum diagonalization with NF sampling. Trains a flow, samples configs, builds projected Hamiltonian, and diagonalizes.

### `SKQDSolver`

```python
SKQDSolver(
    skqd_config: dict | None = None,
    training_config: dict | None = None,
    device: str = "cpu",
)
```

Sample-based Krylov quantum diagonalization with NF sampling and residual expansion.

### `IterativeNFSQDSolver`

```python
IterativeNFSQDSolver(
    n_iterations: int = 30,
    n_samples: int = 5000,
    convergence_tol: float = 1e-6,
    device: str = "cpu",
)
```

Iterative NF-SQD with eigenvector feedback. Trains, diagonalizes, feeds eigenvector back as training target, and repeats.

### `IterativeNFSKQDSolver`

```python
IterativeNFSKQDSolver(
    n_iterations: int = 30,
    n_samples: int = 5000,
    convergence_tol: float = 1e-6,
    device: str = "cpu",
)
```

Iterative NF-SKQD with eigenvector feedback. Same loop as `IterativeNFSQDSolver` but adds Krylov refinement at each iteration.

---

## `qvartools.samplers`

### `SamplerResult`

```python
@dataclass(frozen=True)
class SamplerResult:
    configs: Tensor         # (n_samples, n_sites) sampled configurations
    counts: dict = {}       # Bitstring -> occurrence count
    metadata: dict = {}     # Sampler-specific metadata
```

### `Sampler` (ABC)

```python
class Sampler(ABC):
    @abstractmethod
    def sample(self, n_samples: int) -> SamplerResult: ...
```

### `NFSampler`

```python
NFSampler(
    flow: nn.Module,
    nqs: nn.Module | None = None,
    hamiltonian: Hamiltonian | None = None,
    device: str = "cpu",
)
```

Wraps a trained flow for configuration sampling. Optionally uses NQS for importance weighting.

### `TrotterSampler`

```python
TrotterSampler(
    hamiltonian: Hamiltonian,
    n_trotter_steps: int = 10,
    time_step: float = 0.1,
    initial_state: ndarray | None = None,
)
```

Classical Trotterized time-evolution sampler using `scipy.sparse.linalg.expm_multiply`.

---

## `qvartools.molecules`

### `MOLECULE_REGISTRY`

Dictionary mapping lowercase molecule names to factory metadata. Keys: `h2`, `lih`, `beh2`, `h2o`, `nh3`, `n2`, `ch4`, `c2h4`.

### `get_molecule`

```python
get_molecule(name: str, device: str = "cpu") -> tuple[MolecularHamiltonian, dict]
```

Create a Hamiltonian and info dict for a named molecule. Name is case-insensitive. Raises `KeyError` if not found.

### `list_molecules`

```python
list_molecules() -> list[str]
```

Return sorted list of available molecule names.

---

## `qvartools._utils`

Internal utilities. Not part of the public API but documented for contributors.

### `ConnectionCache`

Hash-based cache for Hamiltonian connections. Avoids recomputing `get_connections` for previously seen configurations.

### `SystemScaler`

Automatic parameter scaling based on system size. Used by `PipelineConfig.adapt_to_system_size`.

### `SystemMetrics`, `ScaledParameters`, `QualityPreset`, `SystemTier`

Supporting classes and enums for the system scaler.

### `gpu_solve_fermion`

GPU-accelerated FCI energy computation via CuPy (when available).

### `configs_to_ibm_format` / `ibm_format_to_configs`

Convert between binary config tensors and IBM SQD bitstring format.

### `vectorized_dedup`

Efficient set-difference deduplication of configuration tensors.

### `expand_basis_via_connections`

Basis expansion via Hamiltonian connections (union of all connected states).

### `hash_config`

Fast integer hash for a single configuration tensor.
