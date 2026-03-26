# ADR-001: Fix SKQD Naming Inaccuracy and Enable NQS Model Interoperability

> **Date:** 2026-03-26
> **Status:** Accepted
> **Deciders:** George Chang (project lead)
> **Handoff target:** Next Claude Code session

---

## Context

During a project review on 2026-03-26, the project lead identified two architectural issues in qvartools that must be resolved before the next release:

### Issue 1: SKQD Naming Inaccuracy

The project contains **two distinct implementations** both loosely called "SKQD":

| Implementation | File | What It Actually Does |
|---|---|---|
| "Classical SKQD" | `src/qvartools/krylov/basis/skqd.py` | **Pure classical linear algebra**: exact matrix exponentiation (`torch.matrix_exp`), Born-rule sampling from exact state vectors (`np.random.default_rng().choice`), dense Hamiltonian matrix. **No quantum circuits, no Trotter decomposition, no Pauli strings.** |
| "Quantum Circuit SKQD" | `src/qvartools/krylov/circuits/circuit_skqd.py` | **Faithful CUDA-Q SKQD implementation**: Trotterized Pauli rotations, `cudaq.sample()` measurement, Pauli-string Hamiltonian projection. References the [NVIDIA CUDA-Q SKQD tutorial](https://nvidia.github.io/cuda-quantum/latest/applications/python/skqd.html). |

The term "SKQD" (Sample-based Krylov Quantum Diagonalization) was introduced by Yu et al. (arXiv:2501.09702, Jan 2025) to describe a **quantum algorithm** where Krylov states are generated via Trotterized quantum circuits and sampled via quantum measurement. The classical implementation in `skqd.py` does none of this — it is a classical subspace method with stochastic basis selection. Calling it "SKQD" is misleading.

**Pipeline routing exacerbates confusion:**
- `subspace_mode="skqd"` routes to the **classical** version (`FlowGuidedSKQD` from `skqd.py`)
- `subspace_mode="skqd_quantum"` routes to the **quantum** version (`QuantumCircuitSKQD` from `circuit_skqd.py`)

The default `subspace_mode` is `"skqd"`, meaning users get the classical version by default, not the real SKQD.

**Directive from project lead:** "SKQD 如果要用就用 CUDA-Q 的，不要用我們寫的 SKQD，那個命名不準確。"

### Issue 2: NQS Model Interoperability

The project has 5 NQS models that are **locked into separate pipelines** due to incompatible interfaces:

| Model | Inherits | NF Training | HI Training |
|---|---|:---:|:---:|
| `DenseNQS` | `NeuralQuantumState` | Yes | No |
| `SignedDenseNQS` | `NeuralQuantumState` | Yes | No |
| `ComplexNQS` | `NeuralQuantumState` | Yes | No |
| `RBMQuantumState` | `NeuralQuantumState` | Yes | No |
| `AutoregressiveTransformer` | `nn.Module` (independent) | No | Yes |

**Root cause: zero interface overlap.**

The NF training pipeline (`PhysicsGuidedFlowTrainer`) consumes NQS exclusively via:
```python
nqs.log_amplitude(x: Tensor[batch, num_sites]) -> Tensor[batch]
```

The HI training pipeline (`run_hi_nqs_sqd`, `run_hi_nqs_skqd`) consumes NQS exclusively via:
```python
nqs.sample(n_samples: int, temperature: float) -> Tensor[n, 2*n_orb]
nqs.log_prob(alpha: Tensor, beta: Tensor) -> Tensor[batch]
```

These are **completely disjoint** method sets. `AutoregressiveTransformer` has no `log_amplitude()`, and `NeuralQuantumState` subclasses have no `sample()` or `log_prob()`.

Additionally, `pipeline.py:_init_components()` hardcodes `DenseNQS` (line 185-188), preventing any other NQS from being used in the main pipeline.

**Directive from project lead:** "這個有辦法讓他們互用嗎？" (Can we make them interoperable?)

---

## Decision

### Decision 1: Rename and Restructure SKQD Implementations

1. **Rename `SampleBasedKrylovDiagonalization`** in `skqd.py` to `ClassicalKrylovDiagonalization` (or similar name that does not imply quantum operations).
2. **Rename `FlowGuidedSKQD`** in `flow_guided.py` to `FlowGuidedKrylovDiag` (or similar).
3. **Make `QuantumCircuitSKQD`** the primary SKQD implementation — when users say "SKQD", they should get the CUDA-Q version.
4. **Update pipeline routing:**
   - `subspace_mode="skqd"` should route to `QuantumCircuitSKQD` (the real SKQD).
   - `subspace_mode="classical_krylov"` (new name) should route to the renamed classical version.
   - Consider keeping `"skqd_quantum"` as an alias for backward compat, but deprecate it.
5. **Update all docstrings, AGENTS.md, README.md, and documentation** to reflect the accurate naming.

### Decision 2: Unify NQS Interface for Cross-Pipeline Interoperability

Create adapter layers so any NQS model can be used in any pipeline:

1. **`TransformerAsNQS` adapter**: Wraps `AutoregressiveTransformer` to expose `log_amplitude(x)` and `phase(x)` for NF training.
   ```python
   class TransformerAsNQS(NeuralQuantumState):
       def log_amplitude(self, x):
           alpha, beta = x[:, :self.n_orb], x[:, self.n_orb:]
           return 0.5 * self.transformer.log_prob(alpha, beta)
       def phase(self, x):
           return torch.zeros(x.shape[0], device=x.device)
   ```

2. **`NQSWithSampling` adapter**: Wraps any `NeuralQuantumState` subclass to expose `sample()` and `log_prob()` for HI training. Sampling could use:
   - The existing `ParticleConservingFlowSampler` as the sampler (already in the project).
   - Or MCMC with the NQS as the proposal distribution.

3. **Make `pipeline.py:_init_components()` configurable**: Accept a `nqs_type` parameter instead of hardcoding `DenseNQS`.

4. **Consider making `AutoregressiveTransformer` inherit from `NeuralQuantumState`**: Add `log_amplitude` and `phase` methods directly, making the transformer a first-class NQS citizen.

### Decision 3: Fix Known Bug in TransformerNFSampler

`transformer_nf_sampler.py:_build_nqs()` line 314 uses `hidden_dim=embed_dim` instead of `hidden_dims=[embed_dim]` when falling back to `DenseNQS`. This will raise `TypeError` at runtime.

---

## Consequences

### What Becomes Easier
- Users get the **real SKQD** (CUDA-Q Trotterized circuits) by default, matching the literature and NVIDIA tutorial.
- All 5 NQS architectures become available to all training pipelines, enabling systematic benchmarking.
- The naming accurately reflects what each component does, reducing confusion for contributors and users.

### What Becomes More Difficult
- **Breaking change**: `subspace_mode="skqd"` will change behavior (from classical to quantum). Existing experiment configs using `"skqd"` will need to be updated to `"classical_krylov"` if they want the old behavior.
- Adapter layers add indirection; performance overhead should be measured.
- `AutoregressiveTransformer.log_prob(alpha, beta) -> log_amplitude(x)` conversion assumes `|psi(x)|^2 ~ p(x)`, which is only an approximation. Document this limitation clearly.

### Follow-up Work Needed
- ADR-002: Decide whether to fully deprecate and remove the classical Krylov implementation, or keep it as a baseline.
- ADR-003: Decide the exact NQS protocol (whether to use Python `Protocol` typing, ABC enforcement, or duck typing).
- Update all 9 experiment scripts in `experiments/methods/` to use the new routing names.
- Update all tests that reference the old class/mode names.

---

## References

### Papers
- **SKQD (original):** Yu et al., "Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization," arXiv:2501.09702 (Jan 2025, v3 Sep 2025). Preprint; not yet journal-published.
- **SQD (original):** Robledo-Moreno et al., "Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer," Science Advances 11, 25, eadu9991 (2025). DOI: 10.1126/sciadv.adu9991.
- **QSCI (equivalent to SQD):** Kanno et al., arXiv:2302.11320 (Feb 2023). QunaSys's independent, earlier formulation of the same algorithm.
- **GenKSR:** arXiv:2512.19420 (Dec 2025). Transformer/Mamba models to learn Krylov sample distributions — related to our NF+SKQD approach.
- **AB-SND:** arXiv:2508.12724 (Aug 2025). Autoregressive NNs replacing quantum sampler in SQD.
- **SqDRIFT:** arXiv:2508.02578 (Aug 2025, rev. Jan 2026). qDRIFT randomized Hamiltonian compilation + SKQD.
- **SKQD accuracy vs classical:** arXiv:2603.03496 (Mar 2026). First demonstration of SKQD outperforming classical sparse solvers on quantum hardware.

### Tools
- **NVIDIA CUDA-Q:** v0.14.0 (released 2026-03-16). SKQD tutorial last updated 2026-03-25 (added multi-GPU eigensolver). https://nvidia.github.io/cuda-quantum/latest/applications/python/skqd.html
- **qiskit-addon-sqd:** v0.12.1 (released 2026-01-16). Also has HPC C++ variant: qiskit-addon-sqd-hpc. https://github.com/Qiskit/qiskit-addon-sqd
- **IBM Quantum Learning:** Dedicated SKQD course module at https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/skqd

### Novelty Assessment (as of 2026-03-26)
**Combining normalizing flows with SKQD (NF-guided SKQD) has no published precedent.** The closest related works are GenKSR (Transformer/Mamba, Dec 2025) and AB-SND (autoregressive NN, Aug 2025). qvartools' NF+SKQD approach is an original contribution direction.

---

## Appendix: Verified Code Evidence

### SKQD Naming (verified 2026-03-26)

| Claim | File:Line | Evidence |
|---|---|---|
| `skqd.py` uses `torch.matrix_exp` | `skqd.py:404` | `self._exp_dt_gpu = torch.matrix_exp(h_gpu)` |
| `skqd.py` uses `rng.choice` for Born-rule sampling | `skqd.py:573-574` | `rng = np.random.default_rng(); indices = rng.choice(...)` |
| `skqd.py` builds dense H matrix | `skqd.py:313-321` | `hamiltonian.matrix_elements(...)` or `to_dense()` |
| `skqd.py` has zero quantum imports | Entire file | No `cudaq`, no Pauli, no gate imports |
| `circuit_skqd.py` references NVIDIA tutorial | `circuit_skqd.py:19-21` | `"NVIDIA CUDA-Q SKQD tutorial: nvidia.github.io/..."` |
| `circuit_skqd.py` uses `cudaq.sample()` | `circuit_skqd.py:768-784` | `result = cudaq.sample(self._kernel_hf, ...)` |
| Pipeline routes "skqd" to classical | `pipeline.py:511` | `else: return self._run_skqd(...)` |
| Pipeline routes "skqd_quantum" to CUDA-Q | `pipeline.py:509` | `elif cfg.subspace_mode == "skqd_quantum"` |
| S matrix always identity | `skqd.py:161,463` | `s_proj = np.eye(n_basis, ...)` |

### NQS Interoperability (verified 2026-03-26)

| Claim | File:Line | Evidence |
|---|---|---|
| AutoregressiveTransformer inherits nn.Module | `autoregressive.py:142` | `class AutoregressiveTransformer(nn.Module)` |
| No `log_amplitude` on Transformer | `autoregressive.py` | Only `log_prob`, `sample`, `forward` defined |
| `log_prob` takes (alpha, beta) | `autoregressive.py:369-373` | `def log_prob(self, alpha: Tensor, beta: Tensor)` |
| NF pipeline calls only `log_amplitude` | `loss_functions.py:150,191,234,294` | All calls are `nqs.log_amplitude(...)` |
| HI pipeline calls `sample` + `log_prob` | `hi_nqs_sqd.py:258,176` | `nqs.sample(...)`, `nqs.log_prob(alpha, beta)` |
| pipeline hardcodes DenseNQS | `pipeline.py:185-188` | `self.nqs = DenseNQS(...)` |
| NeuralQuantumState has no `sample()` | `neural_state.py` | Only `log_amplitude`, `phase`, `log_psi`, `psi`, etc. |
| SignedDenseNQS forces `complex_output=True` | `dense.py:286-289` | `super().__init__(..., complex_output=True)` |
| TransformerNFSampler has DenseNQS fallback bug | `transformer_nf_sampler.py:314` | `DenseNQS(num_sites=n_sites, hidden_dim=embed_dim)` — wrong param name |

### circuit_skqd.py Backend Options (verified 2026-03-26)

| Backend | Description |
|---|---|
| `"auto"` | CUDA-Q if available, else classical fallback |
| `"cudaq"` | Force CUDA-Q circuit simulation |
| `"classical"` | GPU Trotterized state-vector (still has Trotter error, no quantum circuits) |
| `"exact"` | Lanczos exact time evolution (CuPy GPU or CPU) |
| `"lanczos"` | Force Lanczos |
