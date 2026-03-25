# Examples

End-to-end worked examples for qvartools. All examples are designed to be copy-pasted and run directly.

---

## 1. Basic: Compute H2 Ground State Energy

The simplest use case -- compute the ground-state energy of H2 using the full NF-SKQD pipeline with one function call.

```python
from qvartools import run_molecular_benchmark

# Run the full pipeline on H2
results = run_molecular_benchmark("H2", verbose=True)

# Access results
print(f"Final energy:  {results['final_energy']:.10f} Ha")
print(f"Basis size:    {results['skqd_results']['basis_size']}")
print(f"Krylov dim:    {results['skqd_results']['krylov_dim']}")
```

For more control over the pipeline configuration:

```python
from qvartools import PipelineConfig, FlowGuidedKrylovPipeline
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

# Load molecule
hamiltonian, mol_info = get_molecule("H2")

# Compute exact reference
fci = FCISolver()
fci_result = fci.solve(hamiltonian, mol_info)
exact_energy = fci_result.energy
print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")

# Configure and run pipeline
config = PipelineConfig(
    teacher_weight=0.5,
    physics_weight=0.4,
    entropy_weight=0.1,
)

pipeline = FlowGuidedKrylovPipeline(
    hamiltonian=hamiltonian,
    config=config,
    exact_energy=exact_energy,
    auto_adapt=True,
)

results = pipeline.run()
error_mha = results["error_mha"]
print(f"Pipeline energy: {results['final_energy']:.10f} Ha")
print(f"Error: {error_mha:.4f} mHa")
```

---

## 2. Intermediate: Compare SQD vs SKQD on LiH

Compare the SQD solver (no Krylov) with the full SKQD solver on a larger molecule.

```python
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver, SQDSolver, SKQDSolver

# Load LiH (12 qubits)
hamiltonian, mol_info = get_molecule("LiH")

# Exact reference
fci_result = FCISolver().solve(hamiltonian, mol_info)
exact = fci_result.energy
print(f"Exact (FCI): {exact:.10f} Ha")

# SQD solver
sqd_solver = SQDSolver(n_samples=5000, device="cpu")
sqd_result = sqd_solver.solve(hamiltonian, mol_info)
sqd_error = (sqd_result.energy - exact) * 1000
print(f"SQD energy:  {sqd_result.energy:.10f} Ha  (error: {sqd_error:.4f} mHa)")
print(f"SQD basis:   {sqd_result.diag_dim}")
print(f"SQD time:    {sqd_result.wall_time:.2f} s")

# SKQD solver
skqd_solver = SKQDSolver(device="cpu")
skqd_result = skqd_solver.solve(hamiltonian, mol_info)
skqd_error = (skqd_result.energy - exact) * 1000
print(f"SKQD energy: {skqd_result.energy:.10f} Ha  (error: {skqd_error:.4f} mHa)")
print(f"SKQD basis:  {skqd_result.diag_dim}")
print(f"SKQD time:   {skqd_result.wall_time:.2f} s")

# Summary
print(f"\nSKQD improves over SQD by {abs(sqd_error) - abs(skqd_error):.4f} mHa")
```

---

## 3. Advanced: Custom Hamiltonian (Heisenberg Model) with NQS

Use qvartools components directly to study a spin system rather than a molecule.

```python
import torch
from qvartools.hamiltonians import HeisenbergHamiltonian
from qvartools.nqs import DenseNQS
from qvartools.flows import DiscreteFlowSampler, PhysicsGuidedConfig, PhysicsGuidedFlowTrainer
from qvartools.postprocessing import ProjectedHamiltonianBuilder, compute_ground_state_energy

# Create a 10-site antiferromagnetic Heisenberg chain
num_spins = 10
hamiltonian = HeisenbergHamiltonian(
    num_spins=num_spins,
    Jx=1.0,
    Jy=1.0,
    Jz=1.0,
    periodic=True,
)

# Exact ground state for comparison
exact_energy, exact_state = hamiltonian.exact_ground_state()
print(f"Exact ground state energy: {exact_energy:.10f}")

# Create NQS and flow
nqs = DenseNQS(num_sites=num_spins, hidden_dims=[64, 32])
flow = DiscreteFlowSampler(num_sites=num_spins, hidden_dims=[64, 32])

# Configure physics-guided training
training_config = PhysicsGuidedConfig(
    samples_per_batch=200,
    num_batches=5,
    num_epochs=100,
    min_epochs=20,
    teacher_weight=0.5,
    physics_weight=0.4,
    entropy_weight=0.1,
)

# Train
trainer = PhysicsGuidedFlowTrainer(
    flow=flow,
    nqs=nqs,
    hamiltonian=hamiltonian,
    config=training_config,
)
history = trainer.train(progress=True)
print(f"Training epochs: {len(history.get('total_loss', []))}")

# Extract the accumulated basis
basis = trainer.accumulated_basis
print(f"Accumulated {basis.shape[0]} unique configurations")

# Build projected Hamiltonian and solve
builder = ProjectedHamiltonianBuilder(hamiltonian=hamiltonian)
H_proj, S_proj = builder.build(basis.long())

energy = compute_ground_state_energy(H_proj)
error = abs(energy - exact_energy)
print(f"NF-SQD energy: {energy:.10f}")
print(f"Error: {error:.6f}")
```

---

## 4. Advanced: Build a Custom Solver Using qvartools Components

Assemble a custom pipeline from individual components -- train a flow, select diverse configurations, build the projected Hamiltonian, and solve.

```python
import torch
from qvartools.molecules import get_molecule
from qvartools.nqs import DenseNQS
from qvartools.flows import (
    ParticleConservingFlowSampler,
    PhysicsGuidedConfig,
    PhysicsGuidedFlowTrainer,
    verify_particle_conservation,
)
from qvartools.postprocessing import (
    DiversityConfig,
    DiversitySelector,
    ProjectedHamiltonianBuilder,
    solve_generalized_eigenvalue,
)
from qvartools.krylov import (
    ResidualExpansionConfig,
    SelectedCIExpander,
)
from qvartools.solvers import FCISolver

# Load BeH2 (14 qubits)
hamiltonian, mol_info = get_molecule("BeH2")
n_orbitals = hamiltonian.integrals.n_orbitals
n_alpha = hamiltonian.integrals.n_alpha
n_beta = hamiltonian.integrals.n_beta
num_sites = hamiltonian.num_sites

# Exact reference
fci_result = FCISolver().solve(hamiltonian, mol_info)
print(f"Exact energy: {fci_result.energy:.10f} Ha")

# Step 1: Create and train a particle-conserving flow
flow = ParticleConservingFlowSampler(
    num_sites=num_sites,
    n_alpha=n_alpha,
    n_beta=n_beta,
    hidden_dims=[128, 64],
)
nqs = DenseNQS(num_sites=num_sites, hidden_dims=[128, 64])

trainer = PhysicsGuidedFlowTrainer(
    flow=flow,
    nqs=nqs,
    hamiltonian=hamiltonian,
    config=PhysicsGuidedConfig(
        samples_per_batch=500,
        num_batches=10,
        num_epochs=150,
        teacher_weight=0.5,
        physics_weight=0.4,
        entropy_weight=0.1,
    ),
)
history = trainer.train(progress=False)
basis = trainer.accumulated_basis
print(f"Accumulated {basis.shape[0]} configs after training")

# Step 2: Verify particle conservation
is_valid, stats = verify_particle_conservation(
    basis, n_orbitals=n_orbitals, n_alpha=n_alpha, n_beta=n_beta
)
print(f"Particle conservation: {'OK' if is_valid else 'FAILED'}")

# Step 3: Diversity selection
reference = torch.zeros(num_sites)
reference[:n_alpha] = 1.0
reference[n_orbitals:n_orbitals + n_beta] = 1.0

selector = DiversitySelector(
    config=DiversityConfig(max_configs=500, rank_2_fraction=0.4),
    reference=reference,
    n_orbitals=num_sites,
)
selected, sel_stats = selector.select(basis)
print(f"Selected {selected.shape[0]} diverse configs")

# Step 4: Residual expansion
from qvartools.krylov.residual_expansion import _diagonalise_in_basis
energy, eigvec = _diagonalise_in_basis(hamiltonian, selected)
print(f"Pre-expansion energy: {energy:.10f} Ha")

expander = SelectedCIExpander(
    hamiltonian=hamiltonian,
    config=ResidualExpansionConfig(
        max_configs_per_iter=100,
        max_iterations=10,
        residual_threshold=1e-4,
    ),
)
expanded, exp_stats = expander.expand_basis(selected, energy, eigvec)
print(f"Expanded to {expanded.shape[0]} configs")

# Step 5: Final diagonalization
final_energy, _ = _diagonalise_in_basis(hamiltonian, expanded)
error_mha = (final_energy - fci_result.energy) * 1000
print(f"Final energy: {final_energy:.10f} Ha")
print(f"Error: {error_mha:.4f} mHa")
print(f"Chemical accuracy: {'YES' if abs(error_mha) < 1.6 else 'NO'}")
```

---

## 5. Benchmarking: Run All 5 Methods on BeH2

Compare FCI, CCSD, SQD, SKQD, and iterative NF-SQD on the same molecule.

```python
from qvartools.molecules import get_molecule
from qvartools.solvers import (
    FCISolver,
    CCSDSolver,
    SQDSolver,
    SKQDSolver,
    IterativeNFSQDSolver,
)

# Load BeH2
hamiltonian, mol_info = get_molecule("BeH2")
print(f"Molecule: {mol_info['name']} ({mol_info['n_qubits']} qubits)")

# Define solvers
solvers = {
    "FCI": FCISolver(),
    "CCSD": CCSDSolver(),
    "SQD": SQDSolver(n_samples=5000, device="cpu"),
    "SKQD": SKQDSolver(device="cpu"),
    "Iterative NF-SQD": IterativeNFSQDSolver(
        n_iterations=20, n_samples=5000, device="cpu"
    ),
}

# Run all solvers
results = {}
for name, solver in solvers.items():
    print(f"\nRunning {name}...")
    try:
        result = solver.solve(hamiltonian, mol_info)
        results[name] = result
        print(f"  Energy: {result.energy:.10f} Ha")
        print(f"  Time:   {result.wall_time:.2f} s")
        print(f"  Basis:  {result.diag_dim}")
    except Exception as e:
        print(f"  Failed: {e}")

# Comparison table
exact = results["FCI"].energy
print("\n" + "=" * 70)
print(f"{'Method':<22} {'Energy (Ha)':>16} {'Error (mHa)':>12} {'Time (s)':>10}")
print("-" * 70)
for name, result in results.items():
    error = (result.energy - exact) * 1000
    print(f"{name:<22} {result.energy:>16.10f} {error:>12.4f} {result.wall_time:>10.2f}")
print("=" * 70)
```

**Expected output structure** (energies and timings will vary):

```
Method                     Energy (Ha)   Error (mHa)   Time (s)
----------------------------------------------------------------------
FCI                     -15.5959XXXXXX        0.0000       X.XX
CCSD                    -15.5959XXXXXX        X.XXXX       X.XX
SQD                     -15.595XXXXXXX        X.XXXX       X.XX
SKQD                    -15.595XXXXXXX        X.XXXX       X.XX
Iterative NF-SQD        -15.595XXXXXXX        X.XXXX       X.XX
```

Chemical accuracy is 1.6 mHa. Methods achieving error below this threshold are suitable for production quantum chemistry.
