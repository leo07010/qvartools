YAML Configuration System
=========================

qvartools experiment scripts support a three-tier configuration system:

1. **Built-in defaults** — sensible values for all parameters
2. **YAML config files** — reproducible experiment configurations
3. **CLI overrides** — quick parameter adjustments

CLI arguments take precedence over YAML values, which take precedence over
built-in defaults.

Using Config Files
------------------

Each pipeline script in ``experiments/methods/`` accepts a ``--config`` flag:

.. code-block:: bash

   python experiments/methods/flow_ci_krylov.py --config experiments/configs/flow_ci_krylov.yaml

Config files live in ``experiments/configs/`` and are named to match their
pipeline script (e.g., ``flow_ci_krylov.yaml`` for ``flow_ci_krylov.py``).

CLI Overrides
-------------

Any parameter can be overridden on the command line:

.. code-block:: bash

   # Use YAML config but override the molecule and max epochs
   python experiments/methods/flow_ci_krylov.py lih \
       --config experiments/configs/flow_ci_krylov.yaml \
       --max-epochs 200 \
       --teacher-weight 0.6

Available Config Files
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Pipeline
   * - ``flow_ci_krylov.yaml``
     - NF-trained + Direct-CI merged basis -> Krylov expansion
   * - ``flow_ci_sqd.yaml``
     - NF-trained + Direct-CI merged basis -> SQD
   * - ``direct_ci_krylov.yaml``
     - Direct-CI (HF+S+D) -> Krylov (no NF training)
   * - ``direct_ci_sqd.yaml``
     - Direct-CI (HF+S+D) -> SQD
   * - ``iterative_nqs_krylov.yaml``
     - Iterative NQS sampling + Krylov expansion
   * - ``iterative_nqs_sqd.yaml``
     - Iterative NQS sampling + subspace diag
   * - ``flow_only_krylov.yaml``
     - NF-only basis (no CI merge) -> Krylov
   * - ``flow_only_sqd.yaml``
     - NF-only basis (no CI merge) -> SQD
   * - ``hf_only_krylov.yaml``
     - HF-only reference state -> Krylov (baseline)

Config File Structure
---------------------

A typical YAML config file looks like this:

.. code-block:: yaml

   # ---- Molecule -----------------------------------------------
   molecule: h2                  # Molecule identifier

   # ---- Pipeline mode ------------------------------------------
   skip_nf_training: false       # Whether to skip NF training
   subspace_mode: classical_krylov  # classical_krylov, skqd, or sqd

   # ---- Training loss weights ----------------------------------
   teacher_weight: 0.5           # Teacher KL-divergence weight
   physics_weight: 0.4           # Physics-informed energy weight
   entropy_weight: 0.1           # Entropy regularisation weight

   # ---- Training parameters ------------------------------------
   max_epochs: 400               # Maximum training epochs
   min_epochs: 100               # Minimum before early stopping
   samples_per_batch: 2000       # Samples per training batch
   nf_hidden_dims: [256, 256]    # NF network hidden layer sizes
   nqs_hidden_dims: [256, 256, 256, 256]  # NQS hidden sizes

   # ---- SKQD parameters ----------------------------------------
   max_krylov_dim: 15            # Maximum Krylov dimension
   shots_per_krylov: 100000      # Shots per Krylov vector

   # ---- Hardware -----------------------------------------------
   device: auto                  # auto, cpu, or cuda

   # ---- Output -------------------------------------------------
   verbose: true                 # Print progress information

All keys are flat (no nested sections). Keys use underscores and match the
``PipelineConfig`` field names where applicable.

Parameter Reference
-------------------

Common Parameters
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``molecule``
     - ``h2``
     - Molecule identifier (h2, lih, beh2, h2o, nh3, ch4, n2, c2h4)
   * - ``device``
     - ``auto``
     - PyTorch device: ``auto`` (detect GPU), ``cpu``, ``cuda``
   * - ``verbose``
     - ``true``
     - Print detailed progress

Training Parameters
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``teacher_weight``
     - ``0.5``
     - Weight for teacher KL-divergence loss
   * - ``physics_weight``
     - ``0.4``
     - Weight for variational energy loss
   * - ``entropy_weight``
     - ``0.1``
     - Weight for entropy regularization
   * - ``max_epochs``
     - auto-scaled
     - Maximum training epochs
   * - ``min_epochs``
     - auto-scaled
     - Minimum epochs before early stopping
   * - ``samples_per_batch``
     - auto-scaled
     - Samples drawn per training batch

SKQD Parameters
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``max_krylov_dim``
     - auto-scaled
     - Maximum Krylov subspace dimension
   * - ``shots_per_krylov``
     - auto-scaled
     - Shot budget per Krylov vector

SQD Parameters
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``sqd_num_batches``
     - auto-scaled
     - Number of SQD sample batches
   * - ``sqd_self_consistent_iters``
     - ``5``
     - Self-consistent iteration count
   * - ``sqd_noise_rate``
     - auto-scaled
     - Bitflip noise rate for shot simulation

Auto-Scaling
------------

When parameters are not specified in the config file or CLI, qvartools
automatically scales them based on the Hilbert-space size. This auto-scaling
is implemented per-pipeline and uses the number of valid configurations
(determined by the molecule's orbital and electron counts) to choose
appropriate values for training epochs, samples, network sizes, and SKQD/SQD
parameters.

Explicit config values always override auto-scaled defaults.
