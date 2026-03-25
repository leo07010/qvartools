"""Compare solver methods on H2.

Runs FCI, SQD, and SKQD solvers on the same molecule and compares
their accuracy and performance.
"""

from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver, SQDSolver, SKQDSolver

hamiltonian, mol_info = get_molecule("H2")

print(f"Molecule: {mol_info['name']} ({mol_info['n_qubits']} qubits)")
print("=" * 60)

# FCI (exact reference)
fci = FCISolver()
fci_result = fci.solve(hamiltonian, mol_info)
print(f"FCI:  {fci_result.energy:.10f} Ha  ({fci_result.wall_time:.2f}s)")

# SQD
sqd = SQDSolver(n_samples=2000, device="cpu")
sqd_result = sqd.solve(hamiltonian, mol_info)
sqd_error = (sqd_result.energy - fci_result.energy) * 1000
print(f"SQD:  {sqd_result.energy:.10f} Ha  (error: {sqd_error:.4f} mHa, {sqd_result.wall_time:.2f}s)")

# SKQD
skqd = SKQDSolver(device="cpu")
skqd_result = skqd.solve(hamiltonian, mol_info)
skqd_error = (skqd_result.energy - fci_result.energy) * 1000
print(f"SKQD: {skqd_result.energy:.10f} Ha  (error: {skqd_error:.4f} mHa, {skqd_result.wall_time:.2f}s)")
