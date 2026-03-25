# Experiments Guide

This guide covers the experiment pipeline scripts in `experiments/methods/`. Each script implements a single method variant for molecular ground-state energy estimation and can be configured via YAML files with CLI overrides.

---

## Pipeline Overview

All pipelines follow the same pattern:
1. Load a molecule from the registry
2. Compute the exact FCI energy for reference
3. Run the method-specific pipeline
4. Report energy, error vs exact, and timing

### Common Arguments

All scripts accept:
- Positional `molecule` argument (default: `h2`)
- `--config` flag pointing to a YAML configuration file
- `--device` flag (`cpu`, `cuda`, or `auto`)
- Pipeline-specific flags (see `--help`)

---

## 1. `flow_ci_krylov.py` -- NF + Direct-CI -> Krylov Expansion

Trains a normalizing flow, merges NF-sampled basis with Direct-CI (HF + singles + doubles), then runs SKQD Krylov subspace diagonalization.

```bash
python experiments/methods/flow_ci_krylov.py h2
python experiments/methods/flow_ci_krylov.py lih --config experiments/configs/flow_ci_krylov.yaml
```

## 2. `flow_ci_sqd.py` -- NF + Direct-CI -> SQD

Same NF training and basis merge as above, but uses SQD (noise injection + S-CORE batch diagonalization) instead of Krylov.

```bash
python experiments/methods/flow_ci_sqd.py h2
python experiments/methods/flow_ci_sqd.py --config experiments/configs/flow_ci_sqd.yaml
```

## 3. `direct_ci_krylov.py` -- Direct-CI -> Krylov

Skips NF training entirely. Generates HF + singles + doubles deterministically, then applies SKQD Krylov expansion.

```bash
python experiments/methods/direct_ci_krylov.py h2
python experiments/methods/direct_ci_krylov.py --config experiments/configs/direct_ci_krylov.yaml
```

## 4. `direct_ci_sqd.py` -- Direct-CI -> SQD

Direct-CI basis with SQD noise injection and S-CORE. No NF training.

```bash
python experiments/methods/direct_ci_sqd.py h2
python experiments/methods/direct_ci_sqd.py --config experiments/configs/direct_ci_sqd.yaml
```

## 5. `iterative_nqs_krylov.py` -- Iterative NQS + Krylov

Iteratively trains an autoregressive transformer NQS, samples configurations, expands the basis via Hamiltonian connections (Krylov-style), diagonalises in the enlarged subspace, and feeds the eigenvector back as a teacher signal. Repeats until convergence.

```bash
python experiments/methods/iterative_nqs_krylov.py h2
python experiments/methods/iterative_nqs_krylov.py lih --max-iterations 20 --n-samples 3000
```

## 6. `iterative_nqs_sqd.py` -- Iterative NQS + SQD

Same iterative NQS loop as above, but diagonalises directly in the sampled subspace (no Krylov expansion).

```bash
python experiments/methods/iterative_nqs_sqd.py h2
python experiments/methods/iterative_nqs_sqd.py --config experiments/configs/iterative_nqs_sqd.yaml
```

## 7. `flow_only_krylov.py` -- NF-Only -> Krylov (Ablation)

Ablation study: trains NF but uses *only* the NF-generated basis (no Direct-CI merge). Tests how well the flow alone discovers important configurations.

```bash
python experiments/methods/flow_only_krylov.py h2
python experiments/methods/flow_only_krylov.py --config experiments/configs/flow_only_krylov.yaml
```

## 8. `flow_only_sqd.py` -- NF-Only -> SQD (Ablation)

Same NF-only ablation with SQD instead of Krylov.

```bash
python experiments/methods/flow_only_sqd.py h2
python experiments/methods/flow_only_sqd.py --config experiments/configs/flow_only_sqd.yaml
```

## 9. `hf_only_krylov.py` -- HF-Only -> Krylov (Baseline)

Baseline: starts from the single Hartree-Fock reference state with no CI or NF basis. Krylov time evolution discovers configurations through exact propagation.

```bash
python experiments/methods/hf_only_krylov.py h2
python experiments/methods/hf_only_krylov.py --config experiments/configs/hf_only_krylov.yaml
```

---

## Running All Pipelines

```bash
# Quick validation on H2
for script in flow_ci_krylov flow_ci_sqd direct_ci_krylov direct_ci_sqd \
              iterative_nqs_krylov iterative_nqs_sqd \
              flow_only_krylov flow_only_sqd hf_only_krylov; do
    python experiments/methods/${script}.py h2
done
```

## Chemical Accuracy Threshold

All experiments compare results against **1.6 milliHartree (mHa)**, the conventional definition of chemical accuracy.

## Prerequisites

- `pyscf` must be installed for molecular integrals and FCI/CCSD
- GPU experiments require CUDA-enabled PyTorch
- Large molecules (N2, CH4, C2H4) may take several minutes on CPU
