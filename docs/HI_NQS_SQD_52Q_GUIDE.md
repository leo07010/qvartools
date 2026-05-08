# HI-NQS+SQD: 52-qubit-ready guide

A consolidated reference for running **HI-NQS+SQD** (Handover Iterative Neural Quantum State + Sample-based Quantum Diagonalization) on systems up to **52 qubits** (e.g. N₂ CAS(10,26) / cc-pVTZ).

This document captures the full diagnostic path that established what works and what doesn't, the production-grade pipeline (with all performance fixes), and a recipe for new systems. Reference implementation lives in the companion `HI-VQE` repository.

---

## TL;DR

| Pipeline element                          | Status for 52Q | Source code (HI-VQE) |
|-------------------------------------------|----------------|----------------------|
| GPU-batched Davidson on sparse subspace   | ✓ production   | `src/methods/gpu_sparse_det_backend.py` |
| Vectorized GPU Hamiltonian connections    | ✓ production   | `src/hamiltonians/molecular.py` |
| Autoregressive Transformer NQS sampler    | ✓ production   | `src/nqs/transformer.py` |
| PT2 sum-rule selection (default)          | ✓ production   | `src/methods/hi_nqs_sqd.py` |
| HCI heat-bath max-rule selection          | ✓ available    | `_compute_max_h_connection` |
| GPU-resident searchsorted lookup          | ✓ NEW (~440× faster) | `_lookup_coeffs_gpu` |
| Pre-split chunking (no recursive halving) | ✓ NEW          | `_compute_coupling_to_ground_state` |
| Full-S+D enumeration (cross-irrep)        | ✓ NEW          | `MolecularHamiltonian.enable_full_sd_enumeration` |
| c²-eviction (preserve high-amplitude dets) | ✓ NEW          | `c2_eviction` config flag |
| Deep-seed expansion (T/Q from HF)         | ✓ for stretched bonds | `install_deep_seed_expansion` |
| CASSCF orbital optimization (ncas ≥ 15)   | ⚠ REQUIRED for 52Q | `compute_molecular_integrals` (needs patch) |

---

## Why a 52Q-specific guide

Going from 40Q (CAS(10,20), cc-pVTZ) to 52Q (CAS(10,26), cc-pVTZ) exposed structural issues that don't appear at smaller scale:

1. **Hilbert space jumps from 2.4×10⁸ to 6.8×10¹²** (28,000×). Only ~10⁻⁹ of the space is reachable in ≤200k determinants.
2. **Active space orbital choice becomes critical**: at ncas=20 our pipeline used CASCI (HF orbitals) and matched HCI within 0.2 mHa. At ncas=26 the same CASCI fallback costs ~58–68 mHa vs. CASSCF-optimized orbitals. This was misdiagnosed for several iterations as a selection-algorithm bug.
3. **Vectorized chunking breaks under the 8 GB output cap** of `get_connections_vectorized_batch`, leading to recursive halving with exponential wasted work.
4. **CPU-side dict.get lookups in PT2 scoring** dominate wall time (~30 min/iter for full-S+D enumeration before fixes).

The fixes below are essential for 52Q-class problems but also harmless on smaller systems.

---

## Architecture overview

```
                ┌──── NQS sample (Transformer) ────┐
                │                                  │
                ▼                                  ▼
        Candidate pool σ'  ◄──── classical_expansion (top-N basis dets, S+D)
                │
                ▼
        ┌────────── Selection score ──────────┐
        │  PT2 sum:  |Σ_σ H_σσ' c_σ|² / ΔE    │  default
        │  HCI max:  max_σ |H_σσ' · c_σ|       │  optional (--use_hci_score)
        │  NQS log p θ                         │  optional (--use_nqs_score)
        │  γ-MCSCI Krylov moments              │  optional (--use_mcsci_score)
        └─────────────────────────────────────┘
                │ top-K
                ▼
            Subspace S (≤ max_basis_size)
                │
                ▼
        Eviction (when |S| > cap):
          PT2 score (default)  OR  |c|² from prev Davidson (--c2_eviction)
                │
                ▼
        GPU multi-GPU Davidson  →  E_var, c
                │
                ▼
        NQS update on Davidson c   ← teacher signal
                │
                └── next outer iteration
```

12 outer iterations is typical; final **PT2 dressing** (Epstein-Nesbet) is added to E_var to recover external-space correlation.

---

## What we learned at 52Q (chronological)

### 1. Plateau at +71 mHa (variational) on N₂-CAS(10,26)

K=50k → +71.4 mHa above HCI-converged reference (-109.3289 Ha). Bumping K to 200k or 1M gave **no improvement**. Many strategies tested:

| Hypothesis | Test | Result | Conclusion |
|---|---|---|---|
| NQS update bug | RC-T-SCI 3-loss replacement | bit-identical to v4 | NQS isn't the gatekeeper |
| Stale PT2 score in eviction | `c2_eviction` flag | −0.2 mHa | tiny effect |
| Sign-cancellation in PT2 | HCI heat-bath max-rule | +0.04 mHa | not the issue |
| Cross-irrep doubles missed | full-S+D enumeration | +0.06 mHa | symmetry forbids contribution |
| Insufficient initial seed | classical_seed=True (HF+S+D) | 0 mHa | no effect |
| Iterated S+D depth | depth=2 (HF + (S+D)²) | 0 mHa | depth doesn't matter at 52Q |
| **Wrong orbital basis** | **HCI re-run with CASCI** | **HCI now at +81 mHa** | **🟢 found the cause** |

The pipeline used `CASCI` (HF orbitals) for `ncas ≥ 15` (a perf shortcut to avoid CASSCF's internal FCI on huge active spaces). The HCI reference used `CASSCF` (orbital optimization). Apples-to-oranges. **In the same CASCI orbital basis, our pipeline at K=50k was ~10 mHa _better_ than HCI at ε=10⁻³**.

### 2. Stretched bond (R=1.8–2.6 Å) plateau is genuine multi-reference

For 20Q N₂ STO-3G PES:

| R (Å) | NQS+SQD baseline | + deep_seed depth=2 | + deep_seed depth=3 | FCI |
|---|---|---|---|---|
| 1.8 | +110.6 mHa | +0.00 mHa ✓ | +0.00 mHa ✓ | -107.4835 |
| 2.2 | +130.8 mHa | +12.9 mHa | +0.00 mHa ✓ | -107.4449 |
| 2.6 | +125.3 mHa | +2.8 mHa | +0.00 mHa ✓ | -107.4398 |

The diagnosis: NQS sampling biases toward the HF reference; classical_expansion(top-2000) only generates S+D from the current basis. At stretched bonds the dominant configurations are 4-fold (Q) or 6-fold from HF, which never enter the candidate pool. **`install_deep_seed_expansion(depth=N)` directly seeds HF + (S+D)ᴺ into iteration zero**, giving up-to-2N-fold coverage.

For ncas ≤ 26 the seed memory is fine (depth=2 gives ~700k dets, depth=3 hits OOM).

### 3. Cross-irrep determinants are real but contribute zero energy

Diff between HCI ε=10⁻³ basis (44,626 dets, CASSCF orbitals) and v4 K=50k basis showed v4 missing **30,173** dets carrying 4.81 % of |ψ|². Top 30 of these are **all** (α=1, β=1) "spin-coupled doubles" of cross-irrep type, e.g. (3→6_α, 3→5_β) coupling π_x → π_y* with π_x → π_x*.

Direct check: `h2e[6,3,5,3] = 0` *exactly* (different irreps in D₂h). Slater-Condon then gives ⟨HF|H|σ'⟩ = 0, and any single-step path through same-irrep basis dets gives 0 by the same argument. These dets sit in a **decoupled symmetry block** from the singlet ground state and have c=0 in the variational eigenvector. The HCI dump shows c≈0.033 because PySCF's `selected_ci` doesn't enforce strict block-diagonalization; the c values are numerical noise from cross-block leakage and don't affect the ground-state energy.

Empirical confirmation: `enable_full_sd_enumeration()` adds these dets back to the pool; the variational energy at K=50k changes by 0.06 mHa (noise).

---

## Performance fixes (production checklist)

These fixes are required for the pipeline to be usable at 52Q. None changes numerical answers; all are pure performance.

### Fix A: GPU `searchsorted` lookup (`hi_nqs_sqd.py`)

**Problem.** `_compute_coupling_to_ground_state` builds a Python `dict` of basis (a_int, b_int) → c on every recursion level, then loops in Python via `np.fromiter(dict.get(...) for ...)`. For full-SD at 52Q this is ~22 B `dict.get` calls per outer iter ≈ 30 min wall.

**Fix.**
```python
def _build_coeff_lookup(sci_state, n_orb, device=None):
    keys = (a_keys.astype(np.int64) << n_orb) | b_keys.astype(np.int64)
    order = np.argsort(keys)
    sorted_keys = torch.from_numpy(keys[order]).to(device)
    sorted_vals = torch.from_numpy(amps[...].astype(np.float64)).to(device)
    return sorted_keys, sorted_vals

def _lookup_coeffs_gpu(a_ints, b_ints, sorted_keys_t, sorted_vals_t, n_orb):
    query_keys = (a_ints.long() << n_orb) | b_ints.long()
    idx = torch.searchsorted(sorted_keys_t, query_keys).clamp(max=...)
    matches = sorted_keys_t[idx] == query_keys
    return torch.where(matches, sorted_vals_t[idx], 0.0)
```

**Speedup.** Measured: 10⁸ queries × 50k-key sorted array ≈ 7 ms on H100 (≈ 14 G qps). For full-SD 52Q: **30 min → 1.6 s**, ~440× faster.

### Fix B: Cache `_lookup` across recursive halving

**Problem.** When `_compute_coupling_to_ground_state` recursively halves on `MemoryError`, every leaf rebuilds the coefficient lookup (O(N_basis)).

**Fix.** Pass `_lookup` as keyword argument; the top-level call constructs it once and forwards to all leaves.

### Fix C: Pre-split chunking instead of recursive halving

**Problem.** Old behavior: try `get_connections_vectorized_batch(N=1.5M)`, run 41 internal chunks ≈ 40 s, hit 8 GB output cap, raise `MemoryError`. Outer halves to 750k, repeats. After 8 levels of halving, ~10⁴ s of wasted compute compounded across the recursion tree.

**Fix.** Estimate output volume per input config (≈ n_orb² × 432 bytes for full-SD), pick chunk size targeting 16 GB, process linearly with same-size shrink on rare MemoryError:
```python
n_orb_attr = getattr(hamiltonian, "n_orbitals", n_orb)
is_full_sd = getattr(hamiltonian, "enumerate_zero_h", False)
est_conn_per_input = (5 + 2 * n_orb_attr * (n_orb_attr - 1) // 2) if is_full_sd else \
                     max(64, len(hamiltonian._single_p) +
                              len(hamiltonian._double_ab_p) // n_orb_attr)
bytes_per_input = est_conn_per_input * (hamiltonian.num_sites * 8 + 16)
chunk_size = max(1, (16 * 1024**3) // bytes_per_input)

while start < N:
    end = min(start + chunk_size, N)
    coup, ok = _coupling_inner(all_configs[start:end], ...)
    if ok:
        coupling_chunks.append(coup)
        start = end
    else:
        chunk_size = max(1, chunk_size // 2)   # shrink, retry same start
```

### Fix D: Raise output cap from 8 GB to 32 GB (`molecular.py`)

```python
def get_connections_vectorized_batch(self, configs,
                                     max_memory_mb: float = 4096.0,
                                     max_output_mb: float = 32768.0):
```

The H100 nodes have 80 GB GPU memory; 32 GB is comfortable headroom.

### Fix E: GPU-side dedup in `classical_expansion`

`_enumerate_top_amp_connections` previously did `torch.unique(connected.long().cpu(), dim=0)` — moves 30 M × 52-byte tensors to CPU then uniques on CPU.

**Fix.** Pack each row into one `int64` hash (since `n_qubits ≤ 64`), sort + adjacent-difference for uniqueness, all on GPU; only move the deduplicated subset to CPU.

```python
pack_powers = (1 << torch.arange(n_qubits, device=device, dtype=torch.long))
packed_keys = (connected_long * pack_powers).sum(dim=1)
sorted_keys, sort_idx = torch.sort(packed_keys)
is_unique = torch.cat([
    torch.ones(1, dtype=torch.bool, device=device),
    sorted_keys[1:] != sorted_keys[:-1],
])
connected_unique = connected_long[sort_idx[is_unique]].cpu()
```

### Fix F: c²-eviction (optional, default off)

Add a config flag:
```python
@dataclass
class HINQSSQDv3Config(...):
    c2_eviction: bool = False
```

When set, the eviction step uses |c|² from `prev_sci_state` instead of the (possibly stale) PT2 score stored at selection time:

```python
if cfg.c2_eviction and prev_sci_state is not None:
    amps2 = np.abs(prev_sci_state.amplitudes) ** 2
    a_lookup = {int(s): i for i, s in enumerate(prev_sci_state.ci_strs_a)}
    b_lookup = {int(s): i for i, s in enumerate(prev_sci_state.ci_strs_b)}
    all_scores = np.zeros(len(cumulative_bs))
    for i, (a, b) in enumerate(zip(a_ints_eviction, b_ints_eviction)):
        ai, bi = a_lookup.get(int(a)), b_lookup.get(int(b))
        if ai is not None and bi is not None:
            all_scores[i] = float(amps2[ai, bi])
        else:
            all_scores[i] = pt2_scores_raw[i] * scale_to_c2
else:
    all_scores = np.array([cumulative_scores.get(...) for ... in cumulative_bs])
```

Empirically negligible improvement at 52Q (~0.2 mHa) — the value is conceptual: it makes the eviction respect the actual variational importance rather than a stale heat-bath score.

---

## Required orbital choice for 52Q (`molecular.py`)

The single biggest correction at 52Q is using CASSCF-optimized orbitals. The default `compute_molecular_integrals(... casci=True)` for `ncas ≥ 15` was a perf shortcut; for 52Q it costs ~58–68 mHa.

**Patch.** When `ncas ≥ 15`, run `mcscf.CASSCF` with `selected_ci.SCI` as the internal solver:

```python
from pyscf.fci import selected_ci

mc = mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)

if mol.symmetry and mol.topgroup in ("Dooh", "Coov"):
    mc.fcisolver = fci.direct_spin1.FCISolver(mol)

# Selected-CI inside CASSCF for large active spaces
myci = selected_ci.SCI(mol)
myci.select_cutoff = 1e-4
myci.ci_coeff_cutoff = 1e-5
mc.fcisolver = myci
mc.max_cycle_macro = 50
mc.kernel()
```

Tests scheduled to confirm this brings v4 K=50k from +71 mHa down to ~+3 mHa above the converged reference.

---

## Recipe for new molecules / active spaces

```python
from src.molecules import get_molecule
from src.methods.hi_nqs_sqd_v4 import HINQSSQDv4Config, run_hi_nqs_sqd_v4
from src.nqs.transformer import AutoregressiveTransformer

# 1. Build Hamiltonian (use CASSCF orbitals if ncas >= 15; see patch above)
H, info = get_molecule("N2-CAS(10,26)", device="cuda")

# 2. Optional: enable cross-irrep enumeration (only useful for diagnostic;
#    cross-irrep dets contribute 0 to ground-state energy)
# H.enable_full_sd_enumeration()

# 3. For stretched-bond multi-reference, install deep-seed:
# from scratch_n2_strong_corr import install_deep_seed_expansion
# install_deep_seed_expansion(depth=3)   # depth=2 for moderate stretch

# 4. Configure pipeline
cfg = HINQSSQDv4Config(
    n_samples=500_000,            # NQS samples per outer iter
    top_k=20_000,                 # selection top-K
    max_basis_size=50_000,        # final variational basis cap
    max_iterations=12,            # outer iterations
    convergence_threshold=1e-9,
    convergence_window=5,
    nqs_steps=5, nqs_lr=1e-3, entropy_weight=0.05,
    warm_start=False,             # cold-start avoids Davidson lock-in
    classical_seed=False,         # NQS-driven discovery (vs HF+S+D forced)
    classical_expansion=True,
    classical_expansion_top_n=2000,
    final_pt2_correction=True,
    pt2_top_n=10_000,
    use_gpu_sparse_det=True,
    use_gpu_coupling=True,
    c2_eviction=False,            # optional: |c|^2-based eviction
    # Selection score selectors are module-level toggles in
    # src.methods.hi_nqs_sqd._USE_HCI_SCORE[0] = True etc.
)

# 5. Run
result = run_hi_nqs_sqd_v4(H, info, config=cfg)
print(f"E = {result.energy:.6f}  ({len(result.metadata['final_basis']):,} dets)")
```

### Recommended parameters by system size

| n_qubits | Active space | K (max_basis) | n_samples | top_k | walltime (8×H100) |
|---|---|---|---|---|---|
| ≤ 14 | full STO-3G | 500–3 000 | 5 000 | 1 000 | seconds |
| 20–28 | full STO-3G or small CAS | 5 000–25 000 | 50 000 | 5 000 | minutes |
| 40 | CAS(10,20) cc-pVTZ | 50 000–200 000 | 100 000–300 000 | 10 000 | 30–60 min |
| **52** | **CAS(10,26) cc-pVTZ** | **50 000–500 000** | **500 000–1 000 000** | **20 000** | **1–3 h** |

Single seed = 42 is the project convention (single-shot benchmark; statistical variance has been shown to be < 1 mHa across seeds).

---

## SLURM template (large partition)

```bash
#!/bin/bash
#SBATCH --job-name=hi-nqs-52q
#SBATCH --account=<account>
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96       # avoid 1-cpu/cell BLAS oversubscription
#SBATCH --gres=gpu:8             # 8 H100 per node
#SBATCH --mem=0                  # all node RAM (1.9 TB)
#SBATCH --time=05:00:00
#SBATCH --output=logs/hi_nqs_%j.log
#SBATCH --exclude=<known-bad-nodes>

export OMP_NUM_THREADS=16        # critical: prevent oversubscription with 4-8 parallel cells
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/setup_cuda_env.sh
python -u scratch_v4_run.py --gpu=0 --out=results/run.json
```

Key knobs:
- `--cpus-per-task=96` with 8 GPUs gives 12 cpus/cell when running 8 cells in parallel
- `OMP_NUM_THREADS=16` capped to avoid 4×96 = 384 threads contending for 112 cores
- Each cell uses `torch.cuda.set_device(args.gpu)` (passed via `--gpu`); do not modify `CUDA_VISIBLE_DEVICES` (race condition during multi-process init)

---

## Validation suite (used during development)

| Test | Molecule / R | Reference | Pass criterion |
|---|---|---|---|
| Smoke | H₂ STO-3G (4Q) | FCI = -1.137284 | exact match |
| Small mol | LiH/H₂O/N₂ STO-3G | FCI | ≤ 0.001 mHa |
| Medium | C₂H₂ STO-3G (24Q) | HCI ε=10⁻⁶ | ≤ 0.01 mHa at K=25k |
| Stretched MR | N₂ STO-3G R=1.8/2.2/2.6 Å | FCI (14 400 dets) | ≤ 1 mHa with `seed_depth=3` |
| Large CAS | N₂ CAS(10,20) cc-pVTZ (40Q) | HCI ε=10⁻⁴ | < 1 mHa at K=100k |
| **Production** | **N₂ CAS(10,26) cc-pVTZ (52Q)** | **HCI ε=10⁻⁴ (CASSCF)** | **< 5 mHa at K=200k with CASSCF orbitals** |

Numbers without the CASSCF patch will differ by ~58 mHa (the orbital-choice gap).

---

## Known limitations

1. **CASSCF for ncas ≥ 26 takes hours**: PySCF's `selected_ci.SCI` as fcisolver inside CASSCF is feasible but expensive (~30 min for ε=10⁻³ N₂-CAS(10,26)). For ncas ≥ 30 even this becomes prohibitive — would need an external HCI library (e.g. `Dice`).
2. **Multi-reference at strongly-stretched bonds**: `seed_depth=3` works for N₂ at R=2.6 Å but the basis size scaling is roughly `(n_α n_v)^(2N)`. For ncas=26, depth=3 needs > 500 M dets → OOM. A reference-anchored architecture (multiple quasi-references each with their own (S+D)ᴺ neighborhood) is the planned next step.
3. **`enumerate_zero_h=True` was found unhelpful for energy** despite being correct enumeration: cross-irrep dets are decoupled from the singlet block of N₂. Useful for diagnosis only.
4. **PT2 sum-rule + full-S+D combo** had a residual perf bug under heavy candidate-pool loads (Python loop in v3 over ~10⁷ candidates after `_compute_coupling_to_ground_state`). Use HCI heat-bath max-rule (`_USE_HCI_SCORE[0] = True`) when running with `enable_full_sd_enumeration`, or expect 4× slower iter 1.

---

## File map (HI-VQE repository)

```
src/
├── hamiltonians/molecular.py          # CASCI/CASSCF, get_connections, enable_full_sd_enumeration
├── nqs/transformer.py                 # AutoregressiveTransformer (alpha/beta channels with cross-attn)
└── methods/
    ├── hi_nqs_sqd.py                   # Selection scores (PT2/HCI/NQS/MCSCI), _compute_coupling_*
    ├── hi_nqs_sqd_v2.py                # _enumerate_hf_and_connections, _enumerate_top_amp_connections
    ├── hi_nqs_sqd_v3.py                # Main loop with c2_eviction + needs_rescore logic
    ├── hi_nqs_sqd_v4.py                # GPU sparse-det wrapper around v3
    ├── hi_nqs_sqd_v5.py                # Weighted-random expansion + multi-pass PT2 (alternative)
    └── gpu_sparse_det_backend.py       # SparseDetSQDBackend (Davidson)
scratch_n2_strong_corr.py               # install_deep_seed_expansion(depth=N)
scratch_v4_full_sd_52q.py               # full-S+D + c2_eviction + HCI selection driver
scripts/bench_v4_full_sd_52q.slurm      # production SLURM template
```

---

## Provenance

- Initial v3 implementation: PT2-strict version (commit `898375e`).
- v4 GPU sparse-det path and multi-GPU Davidson: HI-VQE branch `feat/gpu-multi-davidson-v4`.
- 52Q diagnostic and fixes (this guide): May 2026, single-seed=42 protocol.
- Reference HCI dump (CASSCF): `scratch_hci_52q_dump.py` with `mcscf.CASSCF + selected_ci.SCI`.

Cross-references:
- 40Q CIPSI/HCI Pareto: `results/n2_40q_cipsi_hci/`
- 52Q HCI dumps (CASSCF and CASCI): `results/hci_52q_dump/`
- 52Q v4 final basis: `results/v4_basis_dump/v4_K50k_basis.npy`
- Stretched N₂ deep-seed: `results/n2_deepseed/`
- All 52Q ablations: `results/c2_eviction_52q/`, `results/cell_c_52q/`, `results/full_sd_52q/`
