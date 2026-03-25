"""Spin Hamiltonian exact diagonalization.

Demonstrates how to construct Heisenberg and transverse-field Ising
Hamiltonians and compute their ground-state energies.
"""

from qvartools.hamiltonians.spin import HeisenbergHamiltonian, TransverseFieldIsing

# Heisenberg model on a 6-site chain
heisenberg = HeisenbergHamiltonian(
    num_spins=6,
    Jx=1.0,
    Jy=1.0,
    Jz=1.0,
    periodic=True,
)

energy, state = heisenberg.exact_ground_state()
print(f"Heisenberg (6 sites, periodic): E0 = {energy:.10f}")

# Transverse-field Ising model
tfim = TransverseFieldIsing(
    num_spins=8,
    V=1.0,
    h=0.5,
    periodic=True,
)

energy, state = tfim.exact_ground_state()
print(f"TFIM (8 sites, h=0.5): E0 = {energy:.10f}")
