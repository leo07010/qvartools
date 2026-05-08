"""HI-NQS+SQD v4 — production-grade pipeline (LiH → 52Q).

A single self-contained sub-package implementing the iterative
Handover Iterative Neural Quantum State + Sample-based Quantum
Diagonalization (HI+NQS+SQD) algorithm with all 52Q-required
performance fixes:

* GPU-resident searchsorted basis lookup (~440× faster than CPU dict.get)
* Pre-split chunking for `_compute_coupling_to_ground_state` (no
  exponentially-wasted work from recursive halving)
* GPU dedup in `_enumerate_top_amp_connections` (sort + adjacent diff)
* `c2_eviction` config flag (preserve high-amplitude dets across iter)
* Optional `enable_full_sd_enumeration` on Hamiltonian to include
  cross-irrep S+D pairs (diagnostic only — they contribute 0 to GS energy)
* `deep_seed.install(depth=N)` for stretched-bond multi-reference
  (HF + (S+D)^N initial basis up to (2N)-fold from HF)

The implementation works for any active-space size; specific tuning per
qubit count is documented in ``docs/HI_NQS_SQD_52Q_GUIDE.md``.

Quick start
-----------
::

    from qvartools.methods.nqs.hi_nqs_v4 import (
        run_hi_nqs_sqd_v4,
        HINQSSQDv4Config,
        MolecularHamiltonian,
        deep_seed,
    )
    from qvartools.methods.nqs.hi_nqs_v4._molecular import compute_molecular_integrals

    integrals = compute_molecular_integrals(
        geometry=[("N", (0, 0, 0)), ("N", (0, 0, 1.10))],
        basis="cc-pvtz",
        cas=(10, 26),
        casci=False,            # use CASSCF orbitals for ncas >= 15
    )
    H = MolecularHamiltonian(integrals, device="cuda")
    info = {"n_qubits": 52, "n_alpha": 5, "n_beta": 5}

    # Optional: stretched-bond multi-reference seeding
    # deep_seed.install(depth=3)

    cfg = HINQSSQDv4Config(
        n_samples=500_000, top_k=20_000, max_basis_size=50_000,
        max_iterations=12, classical_seed=False,
        c2_eviction=False,            # optional |c|² eviction
        use_gpu_sparse_det=True, use_gpu_coupling=True,
    )
    result = run_hi_nqs_sqd_v4(H, info, config=cfg)
    print(f"E = {result.energy:.6f}")
"""

from ._v4 import HINQSSQDv4Config, run_hi_nqs_sqd_v4
from ._v3 import HINQSSQDv3Config, run_hi_nqs_sqd_v3
from ._v2 import HINQSSQDv2Config, run_hi_nqs_sqd_v2
from ._core import (
    HINQSSQDConfig,
    run_hi_nqs_sqd,
    _USE_HCI_SCORE,
    _USE_NQS_SCORE,
    _USE_MCSCI_SCORE,
)
from ._molecular import MolecularHamiltonian, MolecularIntegrals
from ._solver_result import SolverResult, Solver
from ._transformer import AutoregressiveTransformer
from . import deep_seed as deep_seed

__all__ = [
    "HINQSSQDv4Config",
    "run_hi_nqs_sqd_v4",
    "HINQSSQDv3Config",
    "run_hi_nqs_sqd_v3",
    "HINQSSQDv2Config",
    "run_hi_nqs_sqd_v2",
    "HINQSSQDConfig",
    "run_hi_nqs_sqd",
    "MolecularHamiltonian",
    "MolecularIntegrals",
    "SolverResult",
    "Solver",
    "AutoregressiveTransformer",
    "deep_seed",
    "_USE_HCI_SCORE",
    "_USE_NQS_SCORE",
    "_USE_MCSCI_SCORE",
]
