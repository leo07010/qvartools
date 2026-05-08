"""Deep-seed expansion: HF + (S+D)^N seeding for stretched-bond multi-reference.

Replaces the default `_enumerate_hf_and_connections` with depth-N iterated
S+D from HF, generating up to (2*N)-fold excitations directly into the
initial basis. Required for stretched N2 (R >= 1.5 Å) where NQS sampling
cannot reach Q/sextuple excitations on its own.

Usage
-----
    from qvartools.methods.nqs.hi_nqs_v4 import deep_seed
    deep_seed.install(depth=3)             # before run_hi_nqs_sqd_v4

Empirical sizing per depth
--------------------------
    depth=1: HF + S+D       ~  600 dets   (N2 STO-3G, 20Q)
    depth=2: + (S+D)^2     ~ 4 000 dets, includes T+Q
    depth=3: + (S+D)^3     ~10 000 dets, 5/6-fold excitations

For ncas >= 26, depth=3 OOMs (~500M dets).  Use depth=2 only.
"""
from __future__ import annotations

import torch

from . import _v3 as _v3
from ._v2 import _configs_to_ibm_format


def install(depth: int = 2) -> None:
    """Patch the v3 main loop's HF-and-connections enumerator with depth-N
    iterated S+D from HF."""

    def deeper(hamiltonian, n_orb, n_qubits):
        hf = hamiltonian.get_hf_state().to(hamiltonian.device).unsqueeze(0)
        configs = hf.long()
        for d in range(depth):
            connected, _, _ = hamiltonian.get_connections_vectorized_batch(
                configs.float()
            )
            configs = torch.cat([configs, connected.long()], dim=0)
            configs = torch.unique(configs.cpu(), dim=0).to(hamiltonian.device)
            print(f"[deep_seed depth={d+1}] basis size = {configs.shape[0]}",
                  flush=True)
        return _configs_to_ibm_format(configs.long().cpu(), n_orb, n_qubits)

    _v3._enumerate_hf_and_connections = deeper
    print(f"[deep_seed] installed depth={depth} HF expansion", flush=True)


def auto_depth(n_orb: int, n_alpha: int, R: float,
               target_basis_size: int = 10_000) -> int:
    """Heuristic depth selector based on bond length and active-space size.

    Returns the largest N such that the estimated ~ (n_alpha*n_virt)^(2N)
    basis fits within `target_basis_size * 100` dets.
    """
    if R < 1.3:
        return 1
    base_depth = 2 if R < 2.0 else 3
    n_v = max(1, n_orb - n_alpha)
    # Crude lower-bound on basis growth
    estimated_size = (n_alpha * n_v) ** (2 * base_depth)
    if estimated_size > target_basis_size * 100:
        return max(1, base_depth - 1)
    return base_depth
