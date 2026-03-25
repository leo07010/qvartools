"""NF-Trained SKQD --- NF + Direct-CI merged basis -> Krylov expansion.

Runs the 3-stage pipeline:
  1. NF-NQS training (physics-guided mixed-objective)
  2. Diversity-aware basis extraction (merges NF + essential configs)
  3. SKQD Krylov subspace diagonalization

Uses skip_nf_training=False and subspace_mode="skqd".
"""

from __future__ import annotations

import logging
import sys
import time
from math import comb
from pathlib import Path

import torch

# Make the experiments package importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import create_base_parser, load_config  # noqa: E402

from qvartools import FlowGuidedKrylovPipeline, PipelineConfig
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def detect_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_training_params(n_configs: int) -> dict:
    """Scale training hyperparameters based on Hilbert-space size."""
    if n_configs <= 10:
        return dict(max_epochs=100, min_epochs=30, samples_per_batch=500,
                    nf_hidden_dims=[128, 128], nqs_hidden_dims=[128, 128, 128])
    elif n_configs <= 300:
        return dict(max_epochs=150, min_epochs=50, samples_per_batch=1000,
                    nf_hidden_dims=[128, 128], nqs_hidden_dims=[128, 128, 128])
    elif n_configs <= 2000:
        return dict(max_epochs=200, min_epochs=80, samples_per_batch=1500,
                    nf_hidden_dims=[256, 256], nqs_hidden_dims=[256, 256, 256])
    elif n_configs <= 5000:
        return dict(max_epochs=300, min_epochs=100, samples_per_batch=2000,
                    nf_hidden_dims=[256, 256], nqs_hidden_dims=[256, 256, 256])
    else:
        return dict(max_epochs=400, min_epochs=150, samples_per_batch=3000,
                    nf_hidden_dims=[512, 512], nqs_hidden_dims=[512, 512, 512])


def get_skqd_params(n_configs: int) -> dict:
    """Scale SKQD parameters based on Hilbert-space size."""
    if n_configs <= 300:
        return dict(max_krylov_dim=8, shots_per_krylov=100_000)
    elif n_configs <= 5000:
        return dict(max_krylov_dim=10, shots_per_krylov=200_000)
    else:
        return dict(max_krylov_dim=12, shots_per_krylov=200_000)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # --- Parse CLI / YAML config ---
    parser = create_base_parser("NF-Trained SKQD: NF + Direct-CI -> Krylov.")
    parser.add_argument("--teacher-weight", type=float, default=None)
    parser.add_argument("--physics-weight", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--samples-per-batch", type=int, default=None)
    parser.add_argument("--max-krylov-dim", type=int, default=None)
    parser.add_argument("--shots-per-krylov", type=int, default=None)
    parser.add_argument("--max-accumulated-basis", type=int, default=None)
    parser.add_argument("--use-diversity-selection", action="store_true",
                        default=None)
    parser.add_argument("--max-diverse-configs", type=int, default=None)
    parser.add_argument("--use-residual-expansion", action="store_true",
                        default=None)
    parser.add_argument("--residual-iterations", type=int, default=None)
    parser.add_argument("--residual-configs-per-iter", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=None)
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = detect_device()

    # --- Load molecule ---
    hamiltonian, mol_info = get_molecule(args.molecule, device=device)
    n_qubits = mol_info["n_qubits"]
    print(f"Molecule : {mol_info['name']}")
    print(f"Qubits   : {n_qubits}")
    print(f"Basis set: {mol_info['basis']}")
    print(f"Device   : {device}")
    print("=" * 60)

    # --- Compute Hilbert-space size ---
    n_orb = hamiltonian.integrals.n_orbitals
    n_alpha = hamiltonian.integrals.n_alpha
    n_beta = hamiltonian.integrals.n_beta
    n_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
    print(f"Hilbert space: {n_configs:,} configs")

    # --- Compute exact energy for comparison ---
    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")
    print("-" * 60)

    # --- Auto-scale defaults, then override with config values ---
    train_defaults = get_training_params(n_configs)
    skqd_defaults = get_skqd_params(n_configs)

    teacher_weight = config.get("teacher_weight", 0.5)
    physics_weight = config.get("physics_weight", 0.4)
    entropy_weight = config.get("entropy_weight", 0.1)
    max_epochs = config.get("max_epochs", train_defaults["max_epochs"])
    min_epochs = config.get("min_epochs", train_defaults["min_epochs"])
    samples_per_batch = config.get("samples_per_batch",
                                   train_defaults["samples_per_batch"])
    nf_hidden_dims = config.get("nf_hidden_dims",
                                train_defaults["nf_hidden_dims"])
    nqs_hidden_dims = config.get("nqs_hidden_dims",
                                 train_defaults["nqs_hidden_dims"])
    max_krylov_dim = config.get("max_krylov_dim",
                                skqd_defaults["max_krylov_dim"])
    shots_per_krylov = config.get("shots_per_krylov",
                                  skqd_defaults["shots_per_krylov"])

    # --- Build pipeline-config kwargs ---
    pipeline_kwargs: dict = dict(
        skip_nf_training=False,
        subspace_mode="skqd",
        teacher_weight=teacher_weight,
        physics_weight=physics_weight,
        entropy_weight=entropy_weight,
        device=device,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        samples_per_batch=samples_per_batch,
        nf_hidden_dims=nf_hidden_dims,
        nqs_hidden_dims=nqs_hidden_dims,
        max_krylov_dim=max_krylov_dim,
        shots_per_krylov=shots_per_krylov,
    )

    # Optional SKQD knobs -- only pass if explicitly configured
    _optional_keys = [
        "max_accumulated_basis",
        "use_diversity_selection",
        "max_diverse_configs",
        "use_residual_expansion",
        "residual_iterations",
        "residual_configs_per_iter",
    ]
    for key in _optional_keys:
        if key in config:
            pipeline_kwargs[key] = config[key]

    # --- Configure pipeline ---
    pipe_config = PipelineConfig(**pipeline_kwargs)

    # --- Run 3-stage pipeline ---
    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=pipe_config,
        exact_energy=exact_energy,
        auto_adapt=True,
    )

    t_start = time.perf_counter()

    # Stage 1: NF-NQS training
    history = pipeline.train_flow_nqs(progress=True)
    n_epochs = len(history.get("total_loss", []))

    # Stage 2: Basis extraction + diversity selection
    basis = pipeline.extract_and_select_basis()

    # Stage 3: SKQD Krylov diagonalization
    skqd_results = pipeline.run_subspace_diag(progress=True)

    wall_time = time.perf_counter() - t_start

    # --- Results summary ---
    print("\n" + "=" * 60)
    print("NF-TRAINED SKQD RESULTS")
    print("=" * 60)
    print(f"Stage 1 (NF-NQS training):  {n_epochs} epochs")
    print(f"Stage 2 (Basis selection):  {basis.shape[0]} configs")

    if skqd_results.get("energies_per_step"):
        print("\n  SKQD energy convergence:")
        for i, e in enumerate(skqd_results["energies_per_step"]):
            label = "NF-only" if i == 0 else f"k={i - 1}"
            print(f"    {label:>8}: {e:.10f} Ha")

    final_energy = pipeline.results.get("final_energy",
                                        pipeline.results.get("combined_energy"))
    error_mha = pipeline.results.get("error_mha")
    if error_mha is None and final_energy is not None:
        error_mha = (final_energy - exact_energy) * 1000.0

    print(f"\nFinal energy : {final_energy:.10f} Ha")
    print(f"Exact energy : {exact_energy:.10f} Ha")
    if error_mha is not None:
        print(f"Error        : {error_mha:.4f} mHa")
        within = "YES" if abs(error_mha) < CHEMICAL_ACCURACY_MHA else "NO"
        print(f"Chemical acc.: {within} (threshold = {CHEMICAL_ACCURACY_MHA} mHa)")
    print(f"Wall time    : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
