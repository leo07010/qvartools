"""
profile_pipeline --- Wall-clock profiling of the FlowGuidedKrylovPipeline
=========================================================================

Times each pipeline stage (training, basis extraction, subspace diag) and
individual hot-path functions for H2 and BeH2 molecules.  Produces a
formatted bottleneck map at the end.

Usage
-----
    python -m qvartools.experiments.profile_pipeline
    python qvartools/experiments/profile_pipeline.py
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"


def _sync_device() -> None:
    """Call torch.cuda.synchronize() when running on GPU."""
    if HAS_CUDA:
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Timer infrastructure
# ---------------------------------------------------------------------------


@dataclass
class TimingRecord:
    """One recorded timing entry."""

    name: str
    elapsed_s: float
    category: str = ""


class TimingStore:
    """Central store that accumulates all timing records."""

    def __init__(self) -> None:
        self._records: List[TimingRecord] = []

    def add(self, name: str, elapsed_s: float, category: str = "") -> None:
        self._records = [*self._records, TimingRecord(name, elapsed_s, category)]

    @property
    def records(self) -> List[TimingRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records = []

    def records_by_category(self) -> Dict[str, List[TimingRecord]]:
        grouped: Dict[str, List[TimingRecord]] = defaultdict(list)
        for rec in self._records:
            grouped[rec.category].append(rec)
        return dict(grouped)

    def aggregate_by_name(self) -> Dict[str, Tuple[int, float]]:
        """Return {name: (call_count, total_seconds)}."""
        agg: Dict[str, Tuple[int, float]] = {}
        for rec in self._records:
            count, total = agg.get(rec.name, (0, 0.0))
            agg[rec.name] = (count + 1, total + rec.elapsed_s)
        return agg


# Module-level store shared across all helpers
_STORE = TimingStore()


class Timer:
    """Context-manager timer that records elapsed wall-clock time.

    Usage
    -----
    >>> with Timer("my_operation", category="training"):
    ...     expensive_call()
    """

    def __init__(
        self,
        name: str,
        category: str = "",
        store: Optional[TimingStore] = None,
    ) -> None:
        self.name = name
        self.category = category
        self._store = store or _STORE
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        _sync_device()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        _sync_device()
        self.elapsed = time.perf_counter() - self._start
        self._store.add(self.name, self.elapsed, self.category)


# ---------------------------------------------------------------------------
# Function wrapping helpers
# ---------------------------------------------------------------------------


def _make_timed_wrapper(
    original_fn: Callable,
    timing_name: str,
    category: str,
) -> Callable:
    """Return a wrapper that times *original_fn* on every call."""

    @functools.wraps(original_fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer(timing_name, category=category):
            return original_fn(*args, **kwargs)

    # stash the original so we can restore later
    _wrapper._original = original_fn  # type: ignore[attr-defined]
    return _wrapper


def _patch_method(obj: Any, method_name: str, timing_name: str, category: str) -> None:
    """Monkey-patch an instance method with a timed version."""
    original = getattr(obj, method_name, None)
    if original is None:
        print(f"  [WARN] {method_name} not found on {type(obj).__name__}, skipping")
        return
    wrapped = _make_timed_wrapper(original, timing_name, category)
    setattr(obj, method_name, wrapped)


def _patch_module_function(
    module: Any,
    func_name: str,
    timing_name: str,
    category: str,
) -> None:
    """Monkey-patch a module-level function with a timed version."""
    original = getattr(module, func_name, None)
    if original is None:
        print(f"  [WARN] {func_name} not found in {module.__name__}, skipping")
        return
    wrapped = _make_timed_wrapper(original, timing_name, category)
    setattr(module, func_name, wrapped)


def _restore_module_function(module: Any, func_name: str) -> None:
    """Restore the original un-wrapped function on a module."""
    current = getattr(module, func_name, None)
    if current is not None and hasattr(current, "_original"):
        setattr(module, func_name, current._original)


# ---------------------------------------------------------------------------
# Molecule configurations
# ---------------------------------------------------------------------------

MOLECULE_PROFILES: Dict[str, Dict[str, Any]] = {
    "H2": {
        "nf_hidden_dims": [64, 32],
        "nqs_hidden_dims": [64, 32],
        "samples_per_batch": 100,
        "max_epochs": 20,
        "min_epochs": 5,
    },
    "BeH2": {
        "nf_hidden_dims": [128, 64],
        "nqs_hidden_dims": [128, 64],
        "samples_per_batch": 200,
        "max_epochs": 30,
        "min_epochs": 10,
    },
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_time(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 1.0:
        return f"{seconds * 1000:8.2f} ms"
    return f"{seconds:8.3f}  s"


def _print_stage_table(
    molecule: str,
    stage_times: Dict[str, float],
) -> None:
    """Print per-stage timing table for one molecule."""
    total = sum(stage_times.values())
    header = f"  Stage Timing: {molecule}"
    print()
    print(header)
    print("  " + "-" * 58)
    print(f"  {'Stage':<30s} {'Time':>12s} {'%':>7s}")
    print("  " + "-" * 58)
    for stage, elapsed in stage_times.items():
        pct = (elapsed / total * 100.0) if total > 0 else 0.0
        print(f"  {stage:<30s} {_format_time(elapsed):>12s} {pct:6.1f}%")
    print("  " + "-" * 58)
    print(f"  {'TOTAL':<30s} {_format_time(total):>12s} {'100.0':>6s}%")
    print()


def _print_bottleneck_map(store: TimingStore) -> None:
    """Print the top time-consuming functions grouped by subpackage."""
    agg = store.aggregate_by_name()
    if not agg:
        print("  (no function-level timings recorded)")
        return

    # Group by category (subpackage)
    by_cat: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
    for name, (count, total) in agg.items():
        # Derive category from the stored records
        cat = ""
        for rec in store.records:
            if rec.name == name:
                cat = rec.category
                break
        by_cat[cat].append((name, count, total))

    # Sort each category by total time descending
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: x[2], reverse=True)

    grand_total = sum(total for _, total in agg.values())

    print("=" * 72)
    print("  BOTTLENECK MAP: Top time-consuming functions per subpackage")
    print("=" * 72)
    print(
        f"  {'Subpackage':<18s} {'Function':<32s} "
        f"{'Calls':>6s} {'Total':>12s} {'%':>7s}"
    )
    print("  " + "-" * 68)

    for cat in sorted(by_cat.keys()):
        entries = by_cat[cat]
        top_n = entries[:5]  # show top 5 per category
        for func_name, count, total in top_n:
            pct = (total / grand_total * 100.0) if grand_total > 0 else 0.0
            print(
                f"  {cat:<18s} {func_name:<32s} "
                f"{count:>6d} {_format_time(total):>12s} {pct:6.1f}%"
            )
        if len(entries) > 5:
            remaining_time = sum(t for _, _, t in entries[5:])
            remaining_count = sum(c for _, c, _ in entries[5:])
            print(
                f"  {cat:<18s} {'... (' + str(len(entries) - 5) + ' more)':<32s} "
                f"{remaining_count:>6d} {_format_time(remaining_time):>12s}"
            )
    print("  " + "-" * 68)
    print(f"  {'GRAND TOTAL':<50s} {_format_time(grand_total):>12s} 100.0%")
    print()


def _print_comparison_template(all_results: Dict[str, Dict[str, float]]) -> None:
    """Print a before/after timing comparison table template."""
    print("=" * 72)
    print("  BEFORE / AFTER TIMING COMPARISON TEMPLATE")
    print("=" * 72)
    print(
        f"  {'Molecule':<10s} {'Stage':<25s} "
        f"{'Before':>12s} {'After':>12s} {'Speedup':>9s}"
    )
    print("  " + "-" * 68)
    for mol, stages in all_results.items():
        for stage, elapsed in stages.items():
            print(
                f"  {mol:<10s} {stage:<25s} "
                f"{_format_time(elapsed):>12s} {'(TODO)':>12s} {'(TODO)':>9s}"
            )
    print("  " + "-" * 68)
    print()


# ---------------------------------------------------------------------------
# Pipeline profiling
# ---------------------------------------------------------------------------


def _install_function_patches(hamiltonian: Any) -> List[Tuple[Any, str]]:
    """Monkey-patch key functions for per-call timing. Returns restore list."""
    patches_to_restore: List[Tuple[Any, str]] = []

    # --- Hamiltonian methods (instance-level) ---
    ham_methods = [
        ("matrix_elements", "hamiltonians"),
        ("matrix_elements_fast", "hamiltonians"),
        ("get_connections", "hamiltonians"),
        ("diagonal_elements_batch", "hamiltonians"),
    ]
    for method_name, category in ham_methods:
        if hasattr(hamiltonian, method_name):
            _patch_method(hamiltonian, method_name, method_name, category)

    # --- Module-level functions ---
    try:
        from qvartools.flows.training import loss_functions as lf_mod

        _patch_module_function(
            lf_mod, "compute_local_energy", "compute_local_energy", "flows.training"
        )
        patches_to_restore.append((lf_mod, "compute_local_energy"))
    except ImportError:
        print("  [WARN] Could not import loss_functions module")

    try:
        from qvartools.krylov.expansion import residual_config as rc_mod

        _patch_module_function(
            rc_mod,
            "_diagonalise_in_basis",
            "_diagonalise_in_basis",
            "krylov.expansion",
        )
        patches_to_restore.append((rc_mod, "_diagonalise_in_basis"))
    except ImportError:
        print("  [WARN] Could not import residual_config module")

    try:
        from qvartools.krylov.basis import skqd as skqd_mod

        _patch_module_function(
            skqd_mod,
            "_build_projected_matrices",
            "_build_projected_matrices",
            "krylov.basis",
        )
        patches_to_restore.append((skqd_mod, "_build_projected_matrices"))
    except ImportError:
        print("  [WARN] Could not import skqd module")

    try:
        from qvartools.diag.eigen import eigenvalue as ev_mod

        _patch_module_function(
            ev_mod,
            "solve_generalized_eigenvalue",
            "solve_generalized_eigenvalue",
            "diag.eigen",
        )
        patches_to_restore.append((ev_mod, "solve_generalized_eigenvalue"))
    except ImportError:
        print("  [WARN] Could not import eigenvalue module")

    return patches_to_restore


def _remove_function_patches(
    patches: List[Tuple[Any, str]],
) -> None:
    """Restore original module-level functions."""
    for module, func_name in patches:
        _restore_module_function(module, func_name)


def profile_molecule(
    molecule_name: str,
    config_overrides: Dict[str, Any],
) -> Dict[str, float]:
    """Profile the full pipeline for a single molecule.

    Returns a dict of {stage_name: elapsed_seconds}.
    """
    from qvartools.molecules import get_molecule
    from qvartools.pipeline import FlowGuidedKrylovPipeline
    from qvartools.pipeline_config import PipelineConfig

    print(f"\n{'='*72}")
    print(f"  Profiling: {molecule_name}  (device={DEVICE})")
    print(f"{'='*72}")

    # --- Build molecule ---
    with Timer(f"{molecule_name}/get_molecule", category="setup"):
        hamiltonian, mol_info = get_molecule(molecule_name, device=DEVICE)

    print(f"  Molecule loaded: {mol_info.get('name', molecule_name)}")
    print(f"  Qubits: {mol_info.get('num_qubits', '?')}")

    # --- Build config ---
    cfg = PipelineConfig(
        device=DEVICE,
        **config_overrides,
    )

    # --- Build pipeline (without auto_adapt so profiling config is exact) ---
    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=cfg,
        auto_adapt=False,
    )

    # --- Install per-function patches ---
    patches = _install_function_patches(hamiltonian)

    stage_times: Dict[str, float] = {}

    # --- Stage 1: Training ---
    print("  [1/3] Training flow + NQS ...")
    with Timer(f"{molecule_name}/training", category="stage") as t:
        pipeline.train_flow_nqs(progress=False)
    stage_times["Training (flow+NQS)"] = t.elapsed

    # --- Stage 2: Basis extraction ---
    print("  [2/3] Extracting and selecting basis ...")
    with Timer(f"{molecule_name}/basis_extraction", category="stage") as t:
        pipeline.extract_and_select_basis()
    stage_times["Basis extraction"] = t.elapsed

    # --- Stage 3: Subspace diagonalization ---
    print("  [3/3] Running subspace diag ...")
    with Timer(f"{molecule_name}/subspace_diag", category="stage") as t:
        pipeline.run_subspace_diag(progress=False)
    stage_times["Subspace diag"] = t.elapsed

    # --- Clean up patches ---
    _remove_function_patches(patches)

    # --- Print stage breakdown ---
    _print_stage_table(molecule_name, stage_times)

    # --- Print energy result ---
    energy = pipeline.results.get("combined_energy")
    if energy is not None:
        print(f"  Final energy: {energy:.10f} Ha")

    return stage_times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run profiling for all configured molecules and report."""
    print()
    print("#" * 72)
    print("#  qvartools Pipeline Profiler")
    print(f"#  Device: {DEVICE}  |  CUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        print(f"#  GPU: {torch.cuda.get_device_name(0)}")
    print(f"#  PyTorch: {torch.__version__}")
    print("#" * 72)

    _STORE.clear()
    all_stage_results: Dict[str, Dict[str, float]] = {}

    for mol_name, overrides in MOLECULE_PROFILES.items():
        try:
            stage_times = profile_molecule(mol_name, overrides)
            all_stage_results[mol_name] = stage_times
        except Exception as exc:
            print(f"\n  [ERROR] {mol_name} failed: {exc}")
            import traceback

            traceback.print_exc()

    # --- Bottleneck map ---
    print()
    _print_bottleneck_map(_STORE)

    # --- Before/after template ---
    if all_stage_results:
        _print_comparison_template(all_stage_results)

    print("Profiling complete.")


if __name__ == "__main__":
    main()
