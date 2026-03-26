"""NF-Trained SQD --- NF + Direct-CI merged basis -> noise -> S-CORE.

Pipeline: Trains NF-NQS, merges NF samples with essential configs
(HF+S+D), then runs SQD with noise injection and self-consistent
configuration recovery (S-CORE).

Uses skip_nf_training=False with subspace_mode="sqd".
"""

from __future__ import annotations

import logging
import sys
import time
from math import comb
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import create_base_parser, load_config

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
        return dict(
            max_epochs=100,
            min_epochs=30,
            samples_per_batch=500,
            nf_hidden_dims=[128, 128],
            nqs_hidden_dims=[128, 128, 128],
        )
    elif n_configs <= 300:
        return dict(
            max_epochs=150,
            min_epochs=50,
            samples_per_batch=1000,
            nf_hidden_dims=[128, 128],
            nqs_hidden_dims=[128, 128, 128],
        )
    elif n_configs <= 2000:
        return dict(
            max_epochs=200,
            min_epochs=80,
            samples_per_batch=1500,
            nf_hidden_dims=[256, 256],
            nqs_hidden_dims=[256, 256, 256],
        )
    elif n_configs <= 5000:
        return dict(
            max_epochs=300,
            min_epochs=100,
            samples_per_batch=2000,
            nf_hidden_dims=[256, 256],
            nqs_hidden_dims=[256, 256, 256],
        )
    else:
        return dict(
            max_epochs=400,
            min_epochs=150,
            samples_per_batch=3000,
            nf_hidden_dims=[512, 512],
            nqs_hidden_dims=[512, 512, 512],
        )


def get_noise_rate(n_qubits: int) -> float:
    """Return noise rate based on system size."""
    return 0.03 if n_qubits <= 4 else 0.05


def get_shots_multiplier(n_unique: int, n_qubits: int) -> int:
    """Compute shots multiplier targeting ~20K total shots."""
    target_shots = 20_000
    return max(10, min(200, target_shots // max(n_unique, 1)))


def get_sqd_params(n_configs: int) -> dict:
    """Scale SQD parameters based on Hilbert-space size."""
    if n_configs <= 2000:
        sqd_num_batches = 5
    elif n_configs <= 5000:
        sqd_num_batches = 8
    else:
        sqd_num_batches = 10
    return dict(
        sqd_num_batches=sqd_num_batches,
        sqd_self_consistent_iters=5,
        sqd_use_spin_symmetry=n_configs <= 5000,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # --- Parse CLI / YAML config ---
    parser = create_base_parser(
        "NF-Trained SQD: NF + Direct-CI -> noise -> S-CORE.",
    )
    parser.add_argument(
        "--teacher-weight", type=float, default=None, help="Teacher loss weight"
    )
    parser.add_argument(
        "--physics-weight", type=float, default=None, help="Physics loss weight"
    )
    parser.add_argument(
        "--entropy-weight", type=float, default=None, help="Entropy loss weight"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="Maximum training epochs"
    )
    parser.add_argument(
        "--min-epochs", type=int, default=None, help="Minimum training epochs"
    )
    parser.add_argument(
        "--samples-per-batch", type=int, default=None, help="Samples per training batch"
    )
    parser.add_argument(
        "--sqd-num-batches", type=int, default=None, help="Number of SQD batches"
    )
    parser.add_argument(
        "--sqd-self-consistent-iters",
        type=int,
        default=None,
        help="SQD self-consistent iterations",
    )
    parser.add_argument(
        "--sqd-noise-rate", type=float, default=None, help="SQD noise injection rate"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=None, help="Enable verbose output"
    )

    args, cfg = load_config(parser)

    # --- Resolve device ---
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load molecule ---
    hamiltonian, mol_info = get_molecule(cfg.get("molecule", "h2"), device=device)
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

    # --- Auto-scaled defaults ---
    train_defaults = get_training_params(n_configs)
    sqd_defaults = get_sqd_params(n_configs)

    # --- Resolve parameters: config/CLI values -> auto-scaled defaults ---
    max_epochs = cfg.get("max_epochs", train_defaults["max_epochs"])
    min_epochs = cfg.get("min_epochs", train_defaults["min_epochs"])
    samples_per_batch = cfg.get(
        "samples_per_batch", train_defaults["samples_per_batch"]
    )
    nf_hidden_dims = cfg.get("nf_hidden_dims", train_defaults["nf_hidden_dims"])
    nqs_hidden_dims = cfg.get("nqs_hidden_dims", train_defaults["nqs_hidden_dims"])

    teacher_weight = cfg.get("teacher_weight", 0.5)
    physics_weight = cfg.get("physics_weight", 0.4)
    entropy_weight = cfg.get("entropy_weight", 0.1)

    noise_rate = cfg.get("sqd_noise_rate", get_noise_rate(n_qubits))
    sqd_num_batches = cfg.get("sqd_num_batches", sqd_defaults["sqd_num_batches"])
    sqd_self_consistent_iters = cfg.get(
        "sqd_self_consistent_iters", sqd_defaults["sqd_self_consistent_iters"]
    )
    sqd_use_spin_symmetry = cfg.get(
        "sqd_use_spin_symmetry", sqd_defaults["sqd_use_spin_symmetry"]
    )

    print(f"Noise rate: {noise_rate}")
    print(f"SQD batches: {sqd_num_batches}")

    # --- Configure pipeline: NF training + SQD ---
    pipeline_config = PipelineConfig(
        skip_nf_training=False,
        subspace_mode="sqd",
        teacher_weight=teacher_weight,
        physics_weight=physics_weight,
        entropy_weight=entropy_weight,
        sqd_noise_rate=noise_rate,
        device=device,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        samples_per_batch=samples_per_batch,
        nf_hidden_dims=nf_hidden_dims,
        nqs_hidden_dims=nqs_hidden_dims,
        sqd_num_batches=sqd_num_batches,
        sqd_self_consistent_iters=sqd_self_consistent_iters,
        sqd_use_spin_symmetry=sqd_use_spin_symmetry,
    )

    # --- Build pipeline ---
    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=pipeline_config,
        exact_energy=exact_energy,
        auto_adapt=True,
    )

    t_start = time.perf_counter()

    # Stage 1: NF-NQS training
    history = pipeline.train_flow_nqs(progress=True)
    n_epochs = len(history.get("total_loss", []))

    # Stage 2: Basis extraction + diversity selection
    basis = pipeline.extract_and_select_basis()
    n_unique = basis.shape[0]
    print(f"NF+CI merged basis: {n_unique} unique configs")

    # Replicate basis to simulate shot sampling
    shots_mult = get_shots_multiplier(n_unique, n_qubits)
    total_shots = n_unique * shots_mult
    replicated_basis = basis.repeat(shots_mult, 1)
    pipeline.nf_basis = replicated_basis
    print(f"Shots multiplier: {shots_mult}x -> {total_shots} total shots")

    # Stage 3: SQD batch diagonalization with noise + S-CORE
    pipeline.run_subspace_diag(progress=True)

    wall_time = time.perf_counter() - t_start

    # --- Results summary ---
    print("\n" + "=" * 60)
    print("NF-TRAINED SQD RESULTS (NF + Direct-CI -> noise -> S-CORE)")
    print("=" * 60)
    print(f"Stage 1 (NF training): {n_epochs} epochs")
    print(f"Merged basis         : {n_unique} unique configs")
    print(f"Noise rate           : {noise_rate}")
    print(f"Total shots          : {total_shots}")

    final_energy = pipeline.results.get(
        "final_energy", pipeline.results.get("combined_energy")
    )
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
