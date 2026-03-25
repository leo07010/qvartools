"""HI+NQS+SKQD --- Iterative NQS sampling + Krylov expansion + GPU diag.

Iterative pipeline that trains an autoregressive transformer NQS,
samples configurations, expands the basis via Hamiltonian connections
(Krylov-style), and diagonalises in the enlarged subspace.  The
eigenvector is fed back as a teacher signal for the next iteration.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

# Make the experiments package importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import create_base_parser, load_config  # noqa: E402

from qvartools.methods.nqs.hi_nqs_skqd import HINQSSKQDConfig, run_hi_nqs_skqd
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # --- Parse CLI / YAML config ---
    parser = create_base_parser("HI+NQS+SKQD: iterative NQS + Krylov expansion.")
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Maximum outer iterations.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="NQS samples per iteration.",
    )
    parser.add_argument("--verbose", action="store_true", default=None)
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = detect_device()

    # --- Load molecule ---
    hamiltonian, mol_info = get_molecule(args.molecule, device=device)
    print(f"Molecule       : {mol_info['name']}")
    print(f"Qubits         : {mol_info['n_qubits']}")
    print(f"Max iterations : {config.get('max_iterations', 10)}")
    n_samp = config.get("n_samples_per_iter", config.get("n_samples", 5000))
    print(f"Samples/iter   : {n_samp}")
    print("=" * 60)

    # --- Augment mol_info with orbital/electron counts ---
    mol_info["n_orbitals"] = hamiltonian.integrals.n_orbitals
    mol_info["n_alpha"] = hamiltonian.integrals.n_alpha
    mol_info["n_beta"] = hamiltonian.integrals.n_beta

    # --- Exact reference ---
    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    print(f"Exact (FCI) energy: {exact_energy:.10f} Ha\n")

    # --- Configure ---
    config_obj = HINQSSKQDConfig(
        n_iterations=config.get("max_iterations", 10),
        n_samples_per_iter=n_samp,
        nqs_train_epochs=config.get("nqs_train_epochs", 30),
        krylov_max_new=config.get("krylov_max_new", 200),
        krylov_n_ref=config.get("krylov_n_ref", 5),
        device=device,
    )

    # --- Run ---
    t_start = time.perf_counter()
    result = run_hi_nqs_skqd(hamiltonian, mol_info, config=config_obj)
    wall_time = time.perf_counter() - t_start

    # --- Print convergence ---
    energies = result.metadata.get("energy_history", [])
    basis_sizes = result.metadata.get("basis_size_history", [])

    print("\nIteration-by-iteration convergence:")
    print(f"  {'Iter':>4}  {'Energy (Ha)':>16}  {'Error (mHa)':>12}  {'Basis':>8}")
    print("  " + "-" * 50)

    for i, energy in enumerate(energies):
        err_mha = (energy - exact_energy) * 1000.0
        basis = basis_sizes[i] if i < len(basis_sizes) else 0
        print(f"  {i + 1:>4}  {energy:>16.10f}  {err_mha:>12.4f}  {basis:>8d}")

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("HI+NQS+SKQD RESULTS")
    print("=" * 60)
    error_mha = (result.energy - exact_energy) * 1000.0
    within = "YES" if abs(error_mha) < CHEMICAL_ACCURACY_MHA else "NO"
    print(f"Best energy   : {result.energy:.10f} Ha")
    print(f"Exact energy  : {exact_energy:.10f} Ha")
    print(f"Error         : {error_mha:.4f} mHa")
    print(f"Chemical acc. : {within} (threshold = {CHEMICAL_ACCURACY_MHA} mHa)")
    n_iters = result.metadata.get(
        "n_iterations_run", result.metadata.get("n_iterations", 0)
    )
    print(f"Iterations    : {n_iters}")
    print(f"Converged     : {result.converged}")
    print(f"Final basis   : {result.diag_dim}")
    print(f"Wall time     : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
