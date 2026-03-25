Tutorial: Writing a Custom Solver
=================================

This tutorial shows how to implement a custom solver by subclassing the
``Solver`` ABC and returning a ``SolverResult``.

The Solver Interface
--------------------

Every solver must implement one method:

.. code-block:: python

   from qvartools.solvers.solver import Solver, SolverResult
   from qvartools.hamiltonians.hamiltonian import Hamiltonian

   class MySolver(Solver):
       def solve(self, hamiltonian: Hamiltonian, mol_info: dict) -> SolverResult:
           ...

The ``mol_info`` dictionary contains at minimum:

- ``"name"``: molecule name (str)
- ``"n_qubits"``: number of qubits (int)
- ``"basis"``: Gaussian basis set (str)
- ``"geometry"``: atomic coordinates (list)
- ``"charge"``: molecular charge (int)
- ``"spin"``: 2S value (int)

Example: Random Sampling Solver
--------------------------------

Here's a complete example of a solver that samples random configurations,
builds a projected Hamiltonian, and diagonalizes:

.. code-block:: python

   import time
   import numpy as np
   import torch
   from qvartools.solvers.solver import Solver, SolverResult
   from qvartools.hamiltonians.hamiltonian import Hamiltonian
   from qvartools.diag.eigen.projected_hamiltonian import ProjectedHamiltonianBuilder
   from qvartools.diag.eigen.eigensolver import solve_generalized_eigenvalue

   class RandomSamplingSolver(Solver):
       """Solver that samples random valid configurations.

       Parameters
       ----------
       n_samples : int
           Number of random configurations to generate.
       seed : int, optional
           Random seed for reproducibility.
       """

       def __init__(self, n_samples: int = 1000, seed: int | None = None):
           self._n_samples = n_samples
           self._seed = seed

       def solve(self, hamiltonian: Hamiltonian, mol_info: dict) -> SolverResult:
           t0 = time.perf_counter()

           if self._seed is not None:
               torch.manual_seed(self._seed)

           n_sites = hamiltonian.num_sites

           # Generate random binary configurations
           configs = torch.randint(0, 2, (self._n_samples, n_sites))
           configs = torch.unique(configs, dim=0)

           # Build projected Hamiltonian
           builder = ProjectedHamiltonianBuilder(hamiltonian)
           H_proj = builder.build(configs)

           # Diagonalize
           eigenvalues, _ = solve_generalized_eigenvalue(
               H_proj, np.eye(len(H_proj)), k=1
           )

           wall_time = time.perf_counter() - t0
           return SolverResult(
               energy=float(eigenvalues[0]),
               diag_dim=len(configs),
               wall_time=wall_time,
               method="RandomSampling",
               converged=True,
               metadata={"n_samples_requested": self._n_samples},
           )

Using the Custom Solver
-----------------------

.. code-block:: python

   from qvartools.molecules import get_molecule

   hamiltonian, mol_info = get_molecule("H2")

   solver = RandomSamplingSolver(n_samples=500, seed=42)
   result = solver.solve(hamiltonian, mol_info)
   print(result)
   # SolverResult(method='RandomSampling', energy=..., ...)

Design Guidelines
-----------------

When writing custom solvers:

1. **Return ``SolverResult``** with all required fields populated.
2. **Use ``metadata``** for method-specific information (iteration counts,
   per-step energies, convergence history).
3. **Set ``converged``** to ``False`` when the solver reaches its iteration
   limit without meeting the tolerance.
4. **Measure ``wall_time``** from the start of ``solve()`` to the end.
5. **Handle edge cases**: empty basis, singular matrices, missing PySCF, etc.

See :doc:`../developer_guide/extending_solvers` for more advanced patterns.
