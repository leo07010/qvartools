"""Pure SQD --- Direct-CI (HF+S+D) -> noise -> S-CORE -> batch diag.

Pipeline: Generates HF + singles + doubles deterministically (no NF
training), injects shot noise, then runs SQD with self-consistent
configuration recovery (S-CORE) and batch diagonalization.

Uses skip_nf_training=True, subspace_mode="sqd", sqd_noise_rate>0.
After extract_and_select_basis(), replicates basis (shots multiplier),
then run_subspace_diag().
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

    parser = create_base_parser("Pure SQD: Direct-CI (HF+S+D) -> noise -> S-CORE.")
    parser.add_argument("--sqd-num-batches", type=int, default=None,
                        help="Number of SQD batches.")
    parser.add_argument("--sqd-self-consistent-iters", type=int, default=None,
                        help="Number of S-CORE self-consistent iterations.")
    parser.add_argument("--sqd-noise-rate", type=float, default=None,
                        help="Shot noise rate for SQD.")
    parser.add_argument("--verbose", action="store_true", default=None,
                        help="Enable verbose logging.")
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load molecule ---
    hamiltonian, mol_info = get_molecule(config.get("molecule", "h2"), device=device)
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

    # --- Scale parameters (config overrides fallback defaults) ---
    noise_rate = config.get("sqd_noise_rate", get_noise_rate(n_qubits))
    sqd_defaults = get_sqd_params(n_configs)
    sqd_p = {
        "sqd_num_batches": config.get(
            "sqd_num_batches", sqd_defaults["sqd_num_batches"]
        ),
        "sqd_self_consistent_iters": config.get(
            "sqd_self_consistent_iters", sqd_defaults["sqd_self_consistent_iters"]
        ),
        "sqd_use_spin_symmetry": config.get(
            "sqd_use_spin_symmetry", sqd_defaults["sqd_use_spin_symmetry"]
        ),
    }

    print(f"Noise rate: {noise_rate}")
    print(f"SQD batches: {sqd_p['sqd_num_batches']}")

    # --- Configure pipeline: Direct-CI + SQD ---
    pipeline_config = PipelineConfig(
        skip_nf_training=True,
        subspace_mode="sqd",
        sqd_noise_rate=noise_rate,
        device=device,
        **sqd_p,
    )

    # --- Build pipeline ---
    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=pipeline_config,
        exact_energy=exact_energy,
        auto_adapt=True,
    )

    t_start = time.perf_counter()

    # Stage 1: Generate HF+S+D (Direct-CI)
    pipeline.train_flow_nqs(progress=True)

    # Stage 2: Extract basis
    basis = pipeline.extract_and_select_basis()
    n_unique = basis.shape[0]
    print(f"Direct-CI basis: {n_unique} unique configs")

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
    print("PURE SQD RESULTS (Direct-CI -> noise -> S-CORE)")
    print("=" * 60)
    print(f"Direct-CI basis : {n_unique} unique configs")
    print(f"Noise rate      : {noise_rate}")
    print(f"Total shots     : {total_shots}")

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
