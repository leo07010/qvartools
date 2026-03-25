Extending Solvers
=================

This guide explains how to add a new solver to qvartools.

Architecture
------------

All solvers extend the ``Solver`` ABC defined in
``qvartools/solvers/solver.py``. The solver ecosystem follows a flat
hierarchy:

.. code-block:: text

   Solver (ABC)
   ├── FCISolver          (reference/)
   ├── CCSDSolver         (reference/)
   ├── SQDSolver          (subspace/)
   ├── BatchedSQDSolver   (subspace/)
   ├── CIPSISolver        (subspace/)
   ├── SKQDSolver         (krylov/)
   ├── NFSKQDSolver       (krylov/)
   ├── DCISKQDSolver      (krylov/)
   ├── IterativeNFSQDSolver  (iterative/)
   └── IterativeNFSKQDSolver (iterative/)

Step 1: Choose a Location
--------------------------

Place your solver in the appropriate subdirectory:

- ``solvers/reference/`` -- exact or near-exact methods (FCI, CCSD, DMRG)
- ``solvers/subspace/`` -- methods that diagonalize in a sampled subspace
- ``solvers/krylov/`` -- methods that use Krylov subspace techniques
- ``solvers/iterative/`` -- methods with iterative refinement loops

Step 2: Implement the Solver
-----------------------------

.. code-block:: python

   # solvers/subspace/my_solver.py

   from __future__ import annotations

   import time
   from typing import Any, Dict

   from qvartools.hamiltonians.hamiltonian import Hamiltonian
   from qvartools.solvers.solver import Solver, SolverResult


   class MySolver(Solver):
       """My custom subspace solver.

       Parameters
       ----------
       n_samples : int
           Number of configurations to sample.
       tolerance : float
           Convergence tolerance in Hartree.

       Examples
       --------
       >>> from qvartools.molecules import get_molecule
       >>> hamiltonian, mol_info = get_molecule("H2")
       >>> solver = MySolver(n_samples=1000)
       >>> result = solver.solve(hamiltonian, mol_info)
       >>> print(f"Energy: {result.energy:.10f}")
       """

       def __init__(
           self,
           n_samples: int = 1000,
           tolerance: float = 1e-6,
       ) -> None:
           self._n_samples = n_samples
           self._tolerance = tolerance

       def solve(
           self, hamiltonian: Hamiltonian, mol_info: Dict[str, Any]
       ) -> SolverResult:
           t0 = time.perf_counter()

           # Your algorithm here
           energy = self._run_algorithm(hamiltonian)

           return SolverResult(
               energy=energy,
               diag_dim=self._n_samples,
               wall_time=time.perf_counter() - t0,
               method="MySolver",
               converged=True,
           )

       def _run_algorithm(self, hamiltonian: Hamiltonian) -> float:
           ...

Step 3: Register the Solver
----------------------------

Export the solver from its package ``__init__.py``:

.. code-block:: python

   # solvers/subspace/__init__.py
   from qvartools.solvers.subspace.my_solver import MySolver

And from the top-level ``solvers/__init__.py``:

.. code-block:: python

   # solvers/__init__.py
   from qvartools.solvers.subspace import MySolver

Step 4: Add Tests
-----------------

Create a test file in ``tests/test_solvers/``:

.. code-block:: python

   # tests/test_solvers/test_my_solver.py

   import pytest
   from qvartools.solvers.subspace.my_solver import MySolver

   @pytest.mark.pyscf
   def test_my_solver_h2(h2_hamiltonian, h2_mol_info, h2_exact_energy):
       solver = MySolver(n_samples=500)
       result = solver.solve(h2_hamiltonian, h2_mol_info)
       assert result.energy is not None
       assert abs(result.energy - h2_exact_energy) < 0.01  # 10 mHa

Best Practices
--------------

- Use ``dataclasses.replace()`` instead of mutating config objects
- Store iteration history in ``SolverResult.metadata``
- Raise ``ValueError`` for invalid parameters in ``__init__``
- Use ``torch.no_grad()`` for inference-only computation
- Support both CPU and CUDA devices via a ``device`` parameter
