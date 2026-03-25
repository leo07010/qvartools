Diagonalization
===============

.. module:: qvartools.diag

The ``diag`` subpackage provides eigensolvers, diversity selection, and
projected Hamiltonian construction utilities.

Eigensolvers
------------

.. automodule:: qvartools.diag.eigen.eigensolver
   :members:

.. autoclass:: qvartools.diag.eigen.davidson.DavidsonSolver
   :members:
   :show-inheritance:

.. automodule:: qvartools.diag.eigen.eigenvalue
   :members:

Projected Hamiltonian
---------------------

.. autoclass:: qvartools.diag.eigen.projected_hamiltonian.ProjectedHamiltonianBuilder
   :members:
   :show-inheritance:

.. autoclass:: qvartools.diag.eigen.projected_hamiltonian.ProjectedHamiltonianConfig
   :members:

Selection
---------

.. autoclass:: qvartools.diag.selection.diversity_selection.DiversitySelector
   :members:
   :show-inheritance:

.. autoclass:: qvartools.diag.selection.diversity_selection.DiversityConfig
   :members:

.. automodule:: qvartools.diag.selection.excitation_rank
   :members:

.. automodule:: qvartools.diag.selection.bitstring
   :members:
