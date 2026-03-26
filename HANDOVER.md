# HANDOVER.md — Session Handoff for qvartools Refactoring

> **Status: IMPLEMENTATION COMPLETE** (2026-03-26)
> **Branch:** `fix/skqd-naming-and-nqs-interop` (not yet merged to main)
> **Test results:** 240 passed, 0 failed, 34 new tests
>
> **ADR:** `docs/decisions/001-skqd-naming-and-nqs-interoperability.md`
> **Project root:** `/home/mbwcl711/dev/thc1006/qvartools`

---

## Session Log

### 2026-03-26 — Analysis & Decision (Current Session)

**What was done:**
- Full codebase scan: read all 209 files, created `AGENTS.md` (comprehensive project reference)
- Deep analysis of boss's three directives (SKQD naming, CUDA-Q preference, NQS interoperability)
- Web research: verified against 20+ papers, CUDA-Q v0.14.0 docs, qiskit-addon-sqd v0.12.1
- Cross-validated every claim against actual source code (22 claims, all verified TRUE)
- Created ADR-001 at `docs/decisions/001-skqd-naming-and-nqs-interoperability.md`
- Discovered bug in `transformer_nf_sampler.py:314` (`hidden_dim=` should be `hidden_dims=`)

**What was NOT done:**
- No code changes were made. All work was analysis and documentation.
- No tests were run.
- No branches were created.

---

## Current State of the Codebase

- **Branch:** `main` (clean, no uncommitted changes)
- **Last commit:** `34a1381` ��� `docs: rename author to George Chang throughout`
- **CI status:** Assumed passing (not checked this session)

---

## The Three Problems to Fix

### Problem 1: SKQD Naming Inaccuracy

**Summary:** `skqd.py` is pure classical linear algebra (exact matrix_exp + Born-rule sampling) but is named "SKQD" which implies quantum Krylov circuits. The real SKQD is in `circuit_skqd.py`.

**Key files:**
| File | Current Role | Problem |
|---|---|---|
| `src/qvartools/krylov/basis/skqd.py` | Classical Krylov diag | Named "SampleBasedKrylovDiagonalization" — misleading |
| `src/qvartools/krylov/basis/flow_guided.py` | NF-seeded Krylov diag | Named "FlowGuidedSKQD" — misleading |
| `src/qvartools/krylov/circuits/circuit_skqd.py` | Real CUDA-Q SKQD | Correctly named "QuantumCircuitSKQD" |
| `src/qvartools/pipeline.py:507-512` | Routing logic | `"skqd"` routes to classical (wrong default) |
| `src/qvartools/pipeline_config.py:173` | Config | Default `subspace_mode="skqd"` gets classical version |

### Problem 2: NQS Models Cannot Be Used Across Pipelines

**Summary:** Two pipelines speak completely different NQS protocols with zero method overlap.

| Pipeline | NQS Methods Used | Compatible Models |
|---|---|---|
| NF Training (`PhysicsGuidedFlowTrainer`) | `log_amplitude(x)` only | DenseNQS, SignedDenseNQS, ComplexNQS, RBM |
| HI Training (`run_hi_nqs_*`) | `sample(n, temp)` + `log_prob(alpha, beta)` | AutoregressiveTransformer only |
| Main pipeline (`pipeline.py`) | hardcoded `DenseNQS` | DenseNQS only |

### Problem 3: Bug in TransformerNFSampler

**File:** `src/qvartools/samplers/classical/transformer_nf_sampler.py`
**Line:** 314
**Bug:** `DenseNQS(num_sites=n_sites, hidden_dim=embed_dim)` — `hidden_dim` is not a valid parameter; should be `hidden_dims=[embed_dim]`.

---

## TODO List — ALL COMPLETED

### Phase 1: SKQD Naming Fix ✅
- [x] 1.1–1.16: All naming, routing, exports, docstrings, experiments, YAML, docs updated

### Phase 2: NQS Interoperability ✅
- [x] 2.1–2.3: Adapters + `PipelineConfig.nqs_type` + `_create_nqs()` factory
- [x] 2.6–2.8: Tests (including edge cases, memory guard, dtype handling)

### Phase 3: Bug Fix ✅
- [x] 3.1–3.2: `hidden_dim` → `hidden_dims` + regression test

### Phase 4: Documentation & Cleanup ✅
- [x] 4.1–4.5: CHANGELOG, AGENTS.md, all docs, ruff clean, 253 tests passing

### Phase 5: Code Review Fixes ✅
- [x] H2/H3: Source docstrings updated to use new names
- [x] H7: Test for `"skqd"` routing to quantum (mock-based)
- [x] H8: Adapter edge case tests (wrong shape, empty batch, int input, memory guard)
- [x] M1: Unknown `subspace_mode` logs warning
- [x] M2: `NQSWithSampling` memory guard (50K config limit)
- [x] M4: All 8 doc files (6 RST + 2 MD) updated
- [x] M6: `heisenberg_4` fixture deduplicated to `conftest.py`
- [x] L1: `configure_logging` thread-safe via `threading.Lock`
- [x] L2: Invalid log level emits warning
- [x] L3: `log_prob` handles int input via `.float()` conversion

---

## Key Design Decisions for Next Session

### Q: Should we keep backward-compatible aliases for old names?

**Recommendation:** Yes, temporarily. Add deprecated aliases in `__init__.py` files:
```python
# Deprecated aliases (remove in v0.1.0)
SampleBasedKrylovDiagonalization = ClassicalKrylovDiagonalization
FlowGuidedSKQD = FlowGuidedKrylovDiag
```

### Q: Should `AutoregressiveTransformer` directly inherit `NeuralQuantumState`?

**Recommendation:** Use an adapter (`TransformerAsNQS`) instead of modifying the class itself. Reason: the transformer is a **probability model** (outputs `log_prob`), not a **wavefunction model** (outputs `log_amplitude` + `phase`). The `0.5 * log_prob` conversion is an approximation. Mixing these semantics in one class would be confusing.

### Q: What about `circuit_skqd.py`'s 5 backends — does renaming affect them?

**No.** `circuit_skqd.py` (the real SKQD) keeps its name. Only `skqd.py` and `flow_guided.py` get renamed. The `circuit_skqd.py` backends (`auto`, `cudaq`, `classical`, `exact`, `lanczos`) are unaffected.

### Q: Should the default `subspace_mode` change?

**Yes.** After the fix, `subspace_mode="skqd"` should invoke the real CUDA-Q SKQD. However, this requires CUDA-Q to be installed. Consider: if CUDA-Q is not available, `"skqd"` could auto-fallback to `"classical_krylov"` with a warning.

---

## What Failed / Dead Ends

Nothing failed in this session — it was purely analytical. No approaches were attempted and rejected.

---

## External Context the Next Session Should Know

1. **CUDA-Q v0.14.0** was released 2026-03-16. The SKQD tutorial was updated 2026-03-25 (yesterday) with multi-GPU eigensolver support.
2. **qiskit-addon-sqd v0.12.1** is current (2026-01-16). Python 3.9 support was dropped.
3. **NF + SKQD is novel** — no published paper combines normalizing flows with SKQD as of 2026-03-26. Closest: GenKSR (arXiv:2512.19420, Transformer/Mamba), AB-SND (arXiv:2508.12724, autoregressive NN).
4. **QSCI = SQD** — these are the same algorithm with different names from different groups (QunaSys vs IBM).
5. The project uses **Ruff** for linting/formatting. All code must pass `ruff check` before commit.
6. The project uses **Conventional Commits** (`feat:`, `fix:`, `docs:`, `refactor:`).

---

## How to Start the Next Session

```
Read HANDOVER.md and docs/decisions/001-skqd-naming-and-nqs-interoperability.md for full context.
The codebase reference is in AGENTS.md.
Start with Phase 1 of the TODO list (SKQD naming fix).
```
