"""CudaQ SKQD --- HF-only reference state -> Krylov time evolution.

Pipeline: Uses only the HF reference state (1 configuration) as the
starting point. Krylov time evolution discovers additional configurations
through exact time propagation. No singles/doubles, no NF training.

Uses skip_nf_training=True with subspace_mode="classical_krylov". After
train_flow_nqs() generates HF+S+D, we replace _essential_configs with
just the HF state, then extract_and_select_basis(), then run_subspace_diag().
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
    parser = create_base_parser("CudaQ SKQD: HF-only -> Krylov time evolution.")
    parser.add_argument("--max-krylov-dim", type=int, default=None)
    parser.add_argument("--shots-per-krylov", type=int, default=None)
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
    skqd_defaults = get_skqd_params(n_configs)
    max_krylov_dim = config.get("max_krylov_dim",
                                skqd_defaults["max_krylov_dim"])
    shots_per_krylov = config.get("shots_per_krylov",
                                  skqd_defaults["shots_per_krylov"])

    # --- Configure pipeline: Direct-CI mode + SKQD ---
    pipe_config = PipelineConfig(
        skip_nf_training=True,
        subspace_mode="classical_krylov",
        device=device,
        max_krylov_dim=max_krylov_dim,
        shots_per_krylov=shots_per_krylov,
    )

    # --- Build pipeline ---
    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=pipe_config,
        exact_energy=exact_energy,
        auto_adapt=True,
    )

    t_start = time.perf_counter()

    # Stage 1: Direct-CI generates HF+S+D essential configs
    pipeline.train_flow_nqs(progress=True)

    # Override: replace essential configs with HF state only (1 config)
    hf_state = pipeline.reference_state.clone().unsqueeze(0)
    pipeline._essential_configs = hf_state
    print(f"Overriding basis: HF state only ({hf_state.shape[0]} config)")

    # Stage 2: Extract basis (now just the HF state)
    basis = pipeline.extract_and_select_basis()
    print(f"Basis for SKQD: {basis.shape[0]} configs")

    # Stage 3: SKQD Krylov diagonalization discovers configurations
    skqd_results = pipeline.run_subspace_diag(progress=True)

    wall_time = time.perf_counter() - t_start

    # --- Results summary ---
    print("\n" + "=" * 60)
    print("CUDAQ SKQD RESULTS (HF-only -> Krylov)")
    print("=" * 60)
    print("Initial basis: 1 config (HF only)")

    if skqd_results.get("energies_per_step"):
        print("\n  SKQD energy convergence:")
        for i, e in enumerate(skqd_results["energies_per_step"]):
            label = "HF-only" if i == 0 else f"k={i - 1}"
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
