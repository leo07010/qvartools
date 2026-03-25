"""Custom pipeline configuration for LiH.

Shows how to configure the FlowGuidedKrylovPipeline with custom
hyperparameters and run each stage individually.
"""

from qvartools import FlowGuidedKrylovPipeline, PipelineConfig
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

# Load molecule
hamiltonian, mol_info = get_molecule("LiH")

# Compute exact reference energy
fci_result = FCISolver().solve(hamiltonian, mol_info)
print(f"Exact (FCI) energy: {fci_result.energy:.10f} Ha")

# Configure pipeline
config = PipelineConfig(
    teacher_weight=0.5,
    physics_weight=0.4,
    entropy_weight=0.1,
    max_epochs=200,
    min_epochs=50,
    samples_per_batch=1000,
)

pipeline = FlowGuidedKrylovPipeline(
    hamiltonian=hamiltonian,
    config=config,
    exact_energy=fci_result.energy,
    auto_adapt=True,
)

# Run stage by stage
print("\n--- Stage 1: NF-NQS Training ---")
history = pipeline.train_flow_nqs(progress=True)
print(f"Trained for {len(history.get('total_loss', []))} epochs")

print("\n--- Stage 2: Basis Extraction ---")
basis = pipeline.extract_and_select_basis()
print(f"Selected {basis.shape[0]} configurations")

print("\n--- Stage 3: SKQD Diagonalization ---")
skqd_results = pipeline.run_subspace_diag(progress=True)

final_energy = pipeline.results.get("final_energy")
error_mha = (final_energy - fci_result.energy) * 1000.0
print(f"\nFinal energy: {final_energy:.10f} Ha")
print(f"Error: {error_mha:.4f} mHa")
