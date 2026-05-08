# `qvartools.methods.nqs.hi_nqs_v4`

Self-contained sub-package implementing the production-grade **HI-NQS+SQD v4** pipeline that scales from small molecules (LiH, H₂O, ~14Q) all the way to **N₂ CAS(10,26)/cc-pVTZ (52Q)**.

## What's inside

| File | Role |
|---|---|
| `__init__.py` | Public API: `run_hi_nqs_sqd_v4`, configs, `MolecularHamiltonian`, `deep_seed` |
| `_v4.py` | GPU sparse-det + multi-GPU Davidson wrapper around v3 |
| `_v3.py` | Main outer loop; selection / eviction / Davidson / NQS update |
| `_v2.py` | `_enumerate_top_amp_connections` with **GPU-resident dedup** |
| `_core.py` | Selection scores (PT2 / HCI heat-bath / NQS / MCSCI) + **GPU `searchsorted` lookup** + **pre-split chunking** |
| `_molecular.py` | `MolecularHamiltonian` with `enable_full_sd_enumeration()` and `get_connections_vectorized_batch` |
| `_transformer.py` | `AutoregressiveTransformer` (alpha/beta channels, cross-attention) |
| `deep_seed.py` | `install(depth=N)` for stretched-bond multi-reference seeding |
| `_gpu_*.py`, `_sparse_*.py`, `_incremental_sqd.py`, `_multi_gpu_davidson.py` | Davidson / coupling / sparse-det backends |

The underscore prefix on internal modules signals they are implementation details; only the symbols re-exported in `__init__.py` are stable API.

## Quick start (minimum example)

```python
from qvartools.methods.nqs.hi_nqs_v4 import (
    run_hi_nqs_sqd_v4, HINQSSQDv4Config,
    MolecularHamiltonian, deep_seed,
)
from qvartools.methods.nqs.hi_nqs_v4._molecular import compute_molecular_integrals

# 1. Build Hamiltonian
integrals = compute_molecular_integrals(
    geometry=[("N", (0, 0, 0)), ("N", (0, 0, 1.10))],
    basis="cc-pvtz",
    cas=(10, 26),     # active space (ncas, nelecas)
    casci=False,      # use CASSCF orbitals; required for accurate 52Q energies
)
H = MolecularHamiltonian(integrals, device="cuda")
info = {"n_qubits": 52, "n_alpha": 5, "n_beta": 5,
        "n_orbitals": 26, "molecule": "N2-CAS(10,26)"}

# 2. (optional) Stretched-bond multi-reference seed
# deep_seed.install(depth=3)        # only for R >= 1.5 Å

# 3. Configure
cfg = HINQSSQDv4Config(
    n_samples=500_000,
    top_k=20_000,
    max_basis_size=50_000,
    max_iterations=12,
    convergence_threshold=1e-9,
    nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
    classical_seed=False,           # NQS-driven discovery
    classical_expansion=True,
    classical_expansion_top_n=2000,
    final_pt2_correction=True,
    pt2_top_n=10_000,
    use_gpu_sparse_det=True,
    use_gpu_coupling=True,
    c2_eviction=False,              # optional |c|^2 eviction
)

# 4. Run
result = run_hi_nqs_sqd_v4(H, info, config=cfg)
print(f"E = {result.energy:.6f}")
print(f"basis size: {result.metadata['basis_size_history'][-1]:,}")
```

## What's different from `qvartools.methods.nqs.hi_nqs_sqd`

The existing `hi_nqs_sqd.py` is an earlier-design version of the pipeline. This v4 sub-package supersedes it for production work:

- **GPU-resident searchsorted basis lookup** replaces the Python `dict.get` inner loop (~440× faster on H100, critical for full-S+D enumeration)
- **Pre-split chunking** in `_compute_coupling_to_ground_state` replaces recursive halving on `MemoryError` (eliminates exponential wasted work)
- **`c2_eviction`** preserves high-amplitude dets across iterations
- **Multi-GPU Davidson** when `use_multi_gpu_davidson=True`
- **`enable_full_sd_enumeration()`** for diagnostic cross-irrep coverage

The v4 pipeline is also strictly compatible with smaller molecules — fixes are pure performance / configuration; numerical answers match the original v3 for systems that fit comfortably in v3's memory envelope.

## Recommended parameters

| n_qubits | Active space | `max_basis_size` | `n_samples` | `top_k` | walltime (8×H100) |
|---|---|---|---|---|---|
| ≤ 14 | full STO-3G | 500–3 000 | 5 000 | 1 000 | seconds |
| 20–28 | full STO-3G or small CAS | 5 000–25 000 | 50 000 | 5 000 | minutes |
| 40 | CAS(10,20) cc-pVTZ | 50 000–200 000 | 100 000–300 000 | 10 000 | 30–60 min |
| **52** | **CAS(10,26) cc-pVTZ** | **50 000–500 000** | **500 000–1 000 000** | **20 000** | **1–3 h** |

Single seed=42 is the project benchmark convention (variance < 1 mHa across seeds).

## Reference

Full design rationale, diagnostic chronology, and validation suite live in
`docs/HI_NQS_SQD_52Q_GUIDE.md` (top-level of this repository).
