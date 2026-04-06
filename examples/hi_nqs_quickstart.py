"""
HI+NQS+SQD v3 Quickstart
=========================

Minimal example: run HI-NQS v3 on H2O (14 qubits, STO-3G).

Requirements:
    pip install qvartools[quantum]   # includes qiskit-addon-sqd
    # GPU strongly recommended (CUDA)

Usage:
    python examples/hi_nqs_quickstart.py
"""

import torch
from qvartools.molecules import get_molecule
from qvartools.methods.nqs import run_hi_nqs_sqd, HINQSSQDConfig

# ── 1. Load molecule ──────────────────────────────────────────────────────────
H, info = get_molecule("H2O")
print(f"Molecule : H2O — {info['n_qubits']}Q, "
      f"({H.n_alpha}α, {H.n_beta}β) electrons")
print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── 2. Configure HI-NQS v3 ───────────────────────────────────────────────────
cfg = HINQSSQDConfig(
    n_samples=5000,       # NQS samples per iteration
    top_k=500,            # PT2-selected configs kept per iteration
    max_basis_size=0,     # 0 = unlimited basis growth
    max_iterations=30,
    convergence_threshold=1e-8,
    convergence_window=3,
    nf_steps=10,
)

# ── 3. Run ────────────────────────────────────────────────────────────────────
result = run_hi_nqs_sqd(H, info, config=cfg)

# ── 4. Results ────────────────────────────────────────────────────────────────
FCI_REF = -75.0131547015   # H2O STO-3G FCI reference
err_mha = (result.energy - FCI_REF) * 1000

print(f"\n{'='*50}")
print(f"  Energy    : {result.energy:.10f} Ha")
print(f"  Error     : {err_mha:+.4f} mHa  (FCI ref = {FCI_REF:.10f})")
print(f"  Basis size: {result.diag_dim:,} configurations")
print(f"  Converged : {result.converged}")
print(f"  Wall time : {result.wall_time:.1f}s")
print(f"  Chemical accuracy (<1.6 mHa): {'YES ✓' if abs(err_mha) < 1.6 else 'NO'}")
print(f"{'='*50}")
