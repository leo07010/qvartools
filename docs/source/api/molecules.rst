Molecules
=========

.. module:: qvartools.molecules

The ``molecules`` subpackage provides a registry of pre-configured molecular
systems and factory functions.

Registry
--------

.. automodule:: qvartools.molecules.registry
   :members:

Functions
---------

.. autofunction:: qvartools.molecules.get_molecule

.. autofunction:: qvartools.molecules.list_molecules

Available Molecules
-------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - Name
     - Qubits
     - Basis Set
     - Description
   * - H2
     - 4
     - sto-3g
     - Hydrogen molecule (simplest benchmark)
   * - LiH
     - 12
     - sto-6g
     - Lithium hydride
   * - BeH2
     - 14
     - sto-6g
     - Beryllium dihydride
   * - H2O
     - 14
     - sto-6g
     - Water molecule
   * - NH3
     - 16
     - sto-6g
     - Ammonia
   * - CH4
     - 18
     - sto-6g
     - Methane
   * - N2
     - 20
     - cc-pvdz
     - Dinitrogen (strongly correlated)
   * - C2H4
     - 28
     - sto-3g
     - Ethylene (largest benchmark)
