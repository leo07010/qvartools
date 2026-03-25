Installation
============

Requirements
------------

- Python 3.10 or later
- PyTorch 2.0 or later
- NumPy 1.24 or later
- SciPy 1.10 or later

Basic Installation
------------------

Install qvartools in editable mode from the repository root:

.. code-block:: bash

   pip install -e .

This installs the core dependencies (PyTorch, NumPy, SciPy).

Optional Dependencies
---------------------

qvartools provides several optional dependency groups:

.. code-block:: bash

   # PySCF — molecular integrals, RHF, CCSD
   pip install -e ".[pyscf]"

   # Numba — JIT-compiled acceleration for inner loops
   pip install -e ".[numba]"

   # GPU — CuPy for GPU-accelerated eigensolvers
   pip install -e ".[gpu]"

   # Quantum — Qiskit and IBM SQD addon
   pip install -e ".[quantum]"

   # YAML config support
   pip install -e ".[configs]"

   # Full (PySCF + Numba)
   pip install -e ".[full]"

   # Development (pytest, ruff, mypy)
   pip install -e ".[dev]"

   # Documentation (Sphinx + extensions)
   pip install -e ".[docs]"

Using uv
--------

.. code-block:: bash

   uv pip install -e ".[full,dev]"

Docker (GPU)
------------

A GPU-enabled Docker image is provided for environments with NVIDIA GPUs:

.. code-block:: bash

   docker build -f Dockerfile.gpu -t qvartools-gpu .
   docker run --gpus all --rm -it qvartools-gpu

Verifying the Installation
--------------------------

.. code-block:: python

   import qvartools
   print(qvartools.__version__)
   # 0.0.0

To verify PySCF support:

.. code-block:: python

   from qvartools.molecules import list_molecules
   print(list_molecules())
   # ['beh2', 'c2h4', 'ch4', 'h2', 'h2o', 'lih', 'n2', 'nh3']
