Solvers
=======

.. module:: qvartools.solvers

The ``solvers`` subpackage provides high-level solver interfaces that accept
a Hamiltonian and return a standardized :class:`SolverResult`.

Base Classes
------------

.. autoclass:: qvartools.solvers.solver.SolverResult
   :members:

.. autoclass:: qvartools.solvers.solver.Solver
   :members:
   :show-inheritance:

Reference Solvers
-----------------

.. autoclass:: qvartools.solvers.reference.fci.FCISolver
   :members:
   :show-inheritance:

.. autoclass:: qvartools.solvers.reference.ccsd.CCSDSolver
   :members:
   :show-inheritance:

Subspace Solvers
----------------

.. autoclass:: qvartools.solvers.subspace.sqd.SQDSolver
   :members:
   :show-inheritance:

.. autoclass:: qvartools.solvers.subspace.sqd_batched.BatchedSQDSolver
   :members:
   :show-inheritance:

.. autoclass:: qvartools.solvers.subspace.cipsi.CIPSISolver
   :members:
   :show-inheritance:

Krylov Solvers
--------------

.. autoclass:: qvartools.solvers.krylov.skqd.SKQDSolver
   :members:
   :show-inheritance:

.. autoclass:: qvartools.solvers.krylov.nf_skqd.NFSKQDSolver
   :members:
   :show-inheritance:

.. autoclass:: qvartools.solvers.krylov.dci_skqd.DCISKQDSolver
   :members:
   :show-inheritance:

Iterative Solvers
-----------------

.. autoclass:: qvartools.solvers.iterative.iterative_sqd.IterativeNFSQDSolver
   :members:
   :show-inheritance:

.. autoclass:: qvartools.solvers.iterative.iterative_skqd.IterativeNFSKQDSolver
   :members:
   :show-inheritance:
