"""Basic H2 ground-state energy calculation.

Demonstrates the simplest way to use qvartools: load a molecule from the
registry and run the full NF-SKQD pipeline with a single function call.
"""

from qvartools import run_molecular_benchmark

results = run_molecular_benchmark("H2", verbose=True)

print(f"\nGround-state energy: {results['final_energy']:.10f} Ha")
print(f"Basis size: {results['skqd_results']['basis_size']}")
print(f"Krylov dimension: {results['skqd_results']['krylov_dim']}")
