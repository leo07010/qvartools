Hamiltonians
============

.. module:: qvartools.hamiltonians

The ``hamiltonians`` subpackage provides abstract and concrete Hamiltonian
operators for molecular and spin systems.

Base Classes
------------

.. autoclass:: qvartools.hamiltonians.hamiltonian.Hamiltonian
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qvartools.hamiltonians.pauli_string.PauliString
   :members:
   :show-inheritance:

Molecular Hamiltonians
----------------------

.. autoclass:: qvartools.hamiltonians.molecular.hamiltonian.MolecularHamiltonian
   :members:
   :show-inheritance:

.. autoclass:: qvartools.hamiltonians.integrals.MolecularIntegrals
   :members:

.. autofunction:: qvartools.hamiltonians.integrals.compute_molecular_integrals

Spin Hamiltonians
-----------------

.. autoclass:: qvartools.hamiltonians.spin.heisenberg.HeisenbergHamiltonian
   :members:
   :show-inheritance:

.. autoclass:: qvartools.hamiltonians.spin.tfim.TransverseFieldIsing
   :members:
   :show-inheritance:

Jordan-Wigner Mapping
---------------------

.. automodule:: qvartools.hamiltonians.molecular.jordan_wigner
   :members:

Slater-Condon Rules
-------------------

.. automodule:: qvartools.hamiltonians.molecular.slater_condon
   :members:
