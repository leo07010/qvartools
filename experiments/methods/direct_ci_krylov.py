"""Pure SKQD --- Direct-CI (HF+S+D) -> Krylov time evolution.

Pipeline: Generates HF + singles + doubles deterministically (no NF
training), then applies SKQD Krylov subspace diagonalization.

Uses skip_nf_training=True with subspace_mode="classical_krylov". Straight-line:
train_flow_nqs() -> extract_and_select_basis() -> run_subspace_diag().
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

    parser = create_base_parser("Pure SKQD: Direct-CI (HF+S+D) -> Krylov.")
    parser.add_argument(
        "--max-krylov-dim",
        type=int,
        default=None,
        help="Maximum Krylov subspace dimension.",
    )
    parser.add_argument(
        "--shots-per-krylov",
        type=int,
        default=None,
        help="Shots per Krylov expansion step.",
    )
    parser.add_argument(
        "--max-accumulated-basis",
        type=int,
        default=None,
        help="Max accumulated basis size.",
    )
    parser.add_argument(
        "--use-diversity-selection",
        type=bool,
        default=None,
        help="Enable diversity-based selection.",
    )
    parser.add_argument(
        "--max-diverse-configs",
        type=int,
        default=None,
        help="Max diverse configurations to select.",
    )
    parser.add_argument(
        "--use-residual-expansion",
        type=bool,
        default=None,
        help="Enable residual-guided expansion.",
    )
    parser.add_argument(
        "--residual-iterations",
        type=int,
        default=None,
        help="Number of residual expansion iterations.",
    )
    parser.add_argument(
        "--residual-configs-per-iter",
        type=int,
        default=None,
        help="Configs added per residual iteration.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=None, help="Enable verbose logging."
    )
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

    # --- Scale SKQD parameters (config overrides fallback defaults) ---
    skqd_defaults = get_skqd_params(n_configs)
    skqd_p = {
        "max_krylov_dim": config.get("max_krylov_dim", skqd_defaults["max_krylov_dim"]),
        "shots_per_krylov": config.get(
            "shots_per_krylov", skqd_defaults["shots_per_krylov"]
        ),
    }

    # Optional SKQD parameters from config
    for key in (
        "max_accumulated_basis",
        "use_diversity_selection",
        "max_diverse_configs",
        "use_residual_expansion",
        "residual_iterations",
        "residual_configs_per_iter",
    ):
        value = config.get(key)
        if value is not None:
            skqd_p[key] = value

    # --- Configure pipeline: Direct-CI + SKQD ---
    pipeline_config = PipelineConfig(
        skip_nf_training=True,
        subspace_mode="classical_krylov",
        device=device,
        **skqd_p,
    )

    # --- Build and run pipeline ---
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
    print(f"Direct-CI basis: {basis.shape[0]} configs")

    # Stage 3: SKQD Krylov diagonalization
    skqd_results = pipeline.run_subspace_diag(progress=True)

    wall_time = time.perf_counter() - t_start

    # --- Results summary ---
    print("\n" + "=" * 60)
    print("PURE SKQD RESULTS (Direct-CI -> Krylov)")
    print("=" * 60)
    print(f"Direct-CI basis: {basis.shape[0]} configs (HF + singles + doubles)")

    if skqd_results.get("energies_per_step"):
        print("\n  SKQD energy convergence:")
        for i, e in enumerate(skqd_results["energies_per_step"]):
            label = "CI-only" if i == 0 else f"k={i - 1}"
            print(f"    {label:>8}: {e:.10f} Ha")

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
