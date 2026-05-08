"""HI+NQS+SQD v4: v3 + GPU sparse_det + GPU coupling + optional multi-GPU PT2.

Same algorithm as v3 (classical_expansion + final PT2 correction). Heavy-compute
primitives are on GPU:

  - GPUSparseDetSQDBackend: builds H via get_connections_vectorized_batch +
    torch.searchsorted; eigsh on assembled scipy sparse.
  - compute_coupling_gpu: torch.searchsorted replaces CPU dict lookup in
    _compute_coupling_to_ground_state.

Optional single-run multi-GPU (use_multi_gpu=True):
  - Replicates hamiltonian across all visible GPUs.
  - Splits per-iter PT2 candidate scoring across K GPUs (linear speedup).
  - Splits final PT2 external space across K GPUs (linear speedup).
  Requires SLURM to allocate multiple GPUs to the task (e.g. --gres=gpu:4+).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from . import _v3 as _v3
from ._v3 import HINQSSQDv3Config
from ._gpu_sparse_det_backend import GPUSparseDetSQDBackend
from ._gpu_coupling import compute_coupling_gpu


@dataclass
class HINQSSQDv4Config(HINQSSQDv3Config):
    use_gpu_sparse_det: bool = True
    use_gpu_coupling: bool = True
    # Multi-GPU (single-run): replicate hamiltonian across K visible GPUs and
    # split PT2 work. Falls back to single-GPU if only 1 device visible.
    use_multi_gpu: bool = False
    # GPU-native Davidson: replace scipy.sparse.linalg.eigsh (CPU) with torch
    # sparse matvec + Davidson on GPU. Big win when basis > ~100k.
    use_gpu_davidson: bool = False
    # Multi-GPU Davidson: partition H rows across K visible GPUs, parallel matvec.
    # Takes precedence over use_gpu_davidson when both are set.
    use_multi_gpu_davidson: bool = False
    # Multi-GPU NQS sampling: data-parallel across K GPUs (replicated model).
    # Biggest win in v4 where NQS sampling became the bottleneck (~80% of time
    # with n_samples=1M on 40Q). Near-linear speedup expected.
    use_multi_gpu_sampling: bool = False
    multi_gpu_sampling_batch_per_gpu: int = 50_000
    # Multi-GPU on-the-fly Davidson: row-partition basis across K GPUs for the
    # SQD diagonalization step. Required when basis ≥ 1M (single GPU on-the-fly
    # gets too slow / OOMs). Replicates hamiltonian on each GPU once.
    use_multi_gpu_on_the_fly_davidson: bool = False
    on_the_fly_chunk_size: int = 4000
    # SR (Stochastic Reconfiguration) optimizer for NQS — Carleo & Troyer 2017.
    # Replaces Adam with diagonal-Fisher natural gradient. Helps escape mode
    # collapse (especially on 52Q+ where Adam plateaus). Cost: K extra
    # backward passes per training step for Fisher estimate.
    use_sr_optimizer: bool = False
    sr_lr: float = 5e-3
    sr_damping: float = 1e-4
    sr_fisher_K: int = 64
    # MCMC sampling: replace autoregressive forward sample with Metropolis-Hastings
    # chain on |ψ_NQS|². Targets long-tail dets that forward sample misses,
    # which helps escape mode collapse on hard systems (52Q+).
    use_mcmc_sampling: bool = False
    mcmc_n_chains: int = 1024
    mcmc_n_burnin: int = 200
    mcmc_double_frac: float = 0.3
    # NQS architecture override. None = auto-scale by n_orb (v3 default).
    nqs_embed_dim: Optional[int] = None
    nqs_n_heads: Optional[int] = None
    nqs_n_layers: Optional[int] = None
    # Multi-temperature sampling: tuple of T values; per call, sample 1/K at each T
    # and union. Pure NQS-driven exploration: low T captures modes, high T tails.
    multi_temperatures: Optional[tuple] = None


def run_hi_nqs_sqd_v4(hamiltonian, mol_info,
                      config: Optional[HINQSSQDv4Config] = None):
    """Run v3 algorithm with GPU primitives + optional multi-GPU PT2."""
    cfg = config or HINQSSQDv4Config()

    _use_gpu_davidson = cfg.use_gpu_davidson
    _use_multi_gpu_davidson = cfg.use_multi_gpu_davidson

    # If multi-GPU Davidson is active, collect visible devices
    _davidson_devices = None
    if _use_multi_gpu_davidson and torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            _davidson_devices = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
            print(f"    [v4 multi-GPU Davidson] row-partitioning H across {n_gpu} GPUs",
                  flush=True)
        else:
            print(f"    [v4 multi-GPU Davidson] only 1 GPU visible; falling back to single-GPU",
                  flush=True)
            _use_multi_gpu_davidson = False

    # Multi-GPU on-the-fly setup: replicate hamiltonian across all visible GPUs.
    _mg_otf_hams = None
    if cfg.use_multi_gpu_on_the_fly_davidson and torch.cuda.is_available() \
            and torch.cuda.device_count() > 1:
        from ._multi_gpu_pt2 import visible_gpus, replicate_hamiltonian
        devs = visible_gpus()
        print(f"    [v4 multi-GPU on-the-fly] replicating hamiltonian across "
              f"{len(devs)} GPUs ...", flush=True)
        import time as _t
        _t0 = _t.time()
        _mg_otf_hams = replicate_hamiltonian(hamiltonian, devs)
        print(f"    [v4 multi-GPU on-the-fly] replication done "
              f"(t={_t.time()-_t0:.1f}s)", flush=True)

    class _GPUSparseDetFactory:
        def __init__(self, hcore=None, eri=None, n_alpha=None, n_beta=None,
                     spin_sq=None):
            self._backend = GPUSparseDetSQDBackend(
                hamiltonian=hamiltonian,
                n_alpha=n_alpha, n_beta=n_beta, spin_sq=spin_sq,
                use_gpu_davidson=_use_gpu_davidson,
                use_multi_gpu_davidson=_use_multi_gpu_davidson,
                davidson_devices=_davidson_devices,
                use_multi_gpu_on_the_fly=(_mg_otf_hams is not None),
                multi_gpu_hamiltonians=_mg_otf_hams,
            )
        def solve(self, bitstring_matrix):
            return self._backend.solve(bitstring_matrix)

    orig_sparse = _v3.SparseDetSQDBackend
    orig_coupling = _v3._compute_coupling_to_ground_state
    orig_final_pt2 = _v3._final_pt2_correction

    # --- GPU sparse_det ---
    if cfg.use_gpu_sparse_det:
        _v3.SparseDetSQDBackend = _GPUSparseDetFactory

    # --- Multi-GPU setup (requires use_gpu_coupling=True implicitly) ---
    multi_gpu_active = False
    hamiltonians = [hamiltonian]
    if cfg.use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        from ._multi_gpu_pt2 import (
            visible_gpus, replicate_hamiltonian,
            compute_coupling_multi_gpu, final_pt2_multi_gpu,
        )
        devices = visible_gpus()
        print(f"    [v4 multi-GPU] replicating hamiltonian across {len(devices)} GPUs ...",
              flush=True)
        import time as _t
        _t0 = _t.time()
        hamiltonians = replicate_hamiltonian(hamiltonian, devices)
        print(f"    [v4 multi-GPU] replication done (t={_t.time()-_t0:.1f}s)", flush=True)
        multi_gpu_active = True

        # Patch coupling: wrap multi-GPU version with closure over hamiltonians
        def _mg_coupling(candidates, sci_state, _ham_ignored, n_orb, n_qubits):
            return compute_coupling_multi_gpu(
                candidates, sci_state, hamiltonians, n_orb, n_qubits
            )
        _v3._compute_coupling_to_ground_state = _mg_coupling

        # Patch final PT2 with multi-GPU version
        def _mg_final_pt2(cumulative_bs, sci_state, _ham_ignored, n_orb, n_qubits,
                          e0, pt2_top_n, chunk):
            return final_pt2_multi_gpu(
                cumulative_bs, sci_state, hamiltonians,
                n_orb, n_qubits, e0, pt2_top_n, chunk,
            )
        _v3._final_pt2_correction = _mg_final_pt2

    elif cfg.use_gpu_coupling:
        _v3._compute_coupling_to_ground_state = compute_coupling_gpu

    # --- NQS architecture override ---
    _orig_nqs_init = None
    if (cfg.nqs_embed_dim is not None or cfg.nqs_n_heads is not None
            or cfg.nqs_n_layers is not None):
        from ._transformer import AutoregressiveTransformer
        _orig_nqs_init = AutoregressiveTransformer.__init__

        _override_embed = cfg.nqs_embed_dim
        _override_heads = cfg.nqs_n_heads
        _override_layers = cfg.nqs_n_layers

        def _patched_init(self, n_orbitals, n_alpha, n_beta,
                          embed_dim=128, n_heads=4, n_layers=4, **kw):
            ed = _override_embed if _override_embed is not None else embed_dim
            nh = _override_heads if _override_heads is not None else n_heads
            nl = _override_layers if _override_layers is not None else n_layers
            return _orig_nqs_init(
                self, n_orbitals=n_orbitals, n_alpha=n_alpha, n_beta=n_beta,
                embed_dim=ed, n_heads=nh, n_layers=nl, **kw,
            )
        AutoregressiveTransformer.__init__ = _patched_init
        print(f"    [v4 NQS arch override] embed={cfg.nqs_embed_dim}, "
              f"heads={cfg.nqs_n_heads}, layers={cfg.nqs_n_layers}", flush=True)

    # --- Multi-temperature sampling ---
    _orig_sample_for_multi_t = None
    if cfg.multi_temperatures and len(cfg.multi_temperatures) > 0:
        from ._multi_temp_sampler import install_multi_temp_sampler
        _orig_sample_for_multi_t = install_multi_temp_sampler(cfg.multi_temperatures)
        print(f"    [v4 multi-T sampling] T's = {list(cfg.multi_temperatures)}",
              flush=True)

    # --- MCMC sampling for NQS (replaces autoregressive forward) ---
    _orig_sample_for_mcmc = None
    if cfg.use_mcmc_sampling:
        from ._nqs_mcmc_sampler import install_mcmc_sampler
        _orig_sample_for_mcmc = install_mcmc_sampler(
            n_chains=cfg.mcmc_n_chains,
            n_burnin=cfg.mcmc_n_burnin,
            double_frac=cfg.mcmc_double_frac,
        )
        print(f"    [v4 MCMC] using Metropolis-Hastings sampling "
              f"(K={cfg.mcmc_n_chains} chains, burnin={cfg.mcmc_n_burnin}, "
              f"double_frac={cfg.mcmc_double_frac})", flush=True)

    # --- SR optimizer for NQS (replaces Adam) ---
    orig_update_nqs = _v3._update_nqs
    if cfg.use_sr_optimizer:
        from ._nqs_sr_update import update_nqs_sr

        def _sr_update(nqs, optimizer, cumulative_bs, e0, sci_state, hamiltonian,
                       cfg_inner, device, n_orb, n_qubits):
            # `optimizer` (Adam) is ignored under SR
            update_nqs_sr(
                nqs,
                sr_lr=cfg.sr_lr, sr_damping=cfg.sr_damping,
                sr_fisher_K=cfg.sr_fisher_K,
                cumulative_bs=cumulative_bs, e0=e0,
                sci_state=sci_state, hamiltonian=hamiltonian,
                cfg=cfg_inner, device=device,
                n_orb=n_orb, n_qubits=n_qubits,
            )
        _v3._update_nqs = _sr_update
        print(f"    [v4 SR] using Stochastic Reconfiguration optimizer "
              f"(lr={cfg.sr_lr}, damping={cfg.sr_damping}, K={cfg.sr_fisher_K})",
              flush=True)

    # --- Multi-GPU NQS sampling ---
    _sampling_state = None
    if (cfg.use_multi_gpu_sampling
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1):
        from ._multi_gpu_nqs_sampler import install_multi_gpu_sampling
        devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        print(f"    [v4 multi-GPU sampling] data-parallel NQS across {len(devs)} GPUs",
              flush=True)
        _sampling_state = install_multi_gpu_sampling(
            devs, batch_per_gpu=cfg.multi_gpu_sampling_batch_per_gpu
        )

    try:
        result = _v3.run_hi_nqs_sqd_v3(hamiltonian, mol_info, config=cfg)
    finally:
        _v3.SparseDetSQDBackend = orig_sparse
        _v3._compute_coupling_to_ground_state = orig_coupling
        _v3._final_pt2_correction = orig_final_pt2
        _v3._update_nqs = orig_update_nqs
        if _sampling_state is not None:
            from ._multi_gpu_nqs_sampler import uninstall_multi_gpu_sampling
            uninstall_multi_gpu_sampling(_sampling_state)
        if _orig_sample_for_mcmc is not None:
            from ._nqs_mcmc_sampler import uninstall_mcmc_sampler
            uninstall_mcmc_sampler(_orig_sample_for_mcmc)
        if _orig_nqs_init is not None:
            from ._transformer import AutoregressiveTransformer
            AutoregressiveTransformer.__init__ = _orig_nqs_init
        if _orig_sample_for_multi_t is not None:
            from ._multi_temp_sampler import uninstall_multi_temp_sampler
            uninstall_multi_temp_sampler(_orig_sample_for_multi_t)

    result.method = "HI+NQS+SQD-v4"
    result.metadata["v4_gpu_sparse_det"] = cfg.use_gpu_sparse_det
    result.metadata["v4_gpu_coupling"] = cfg.use_gpu_coupling
    result.metadata["v4_multi_gpu"] = multi_gpu_active
    result.metadata["v4_n_gpus_used"] = len(hamiltonians) if multi_gpu_active else 1
    return result
