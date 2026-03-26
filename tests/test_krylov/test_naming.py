"""Tests for SKQD naming refactoring (ADR-001).

These tests verify:
1. New class names exist and work (ClassicalKrylovDiagonalization, FlowGuidedKrylovDiag)
2. Deprecated aliases still resolve to the new classes
3. Pipeline routing maps "classical_krylov" to the classical version
4. Pipeline routing maps "skqd" to the real CUDA-Q SKQD (with fallback)
5. Logging emits deprecation warnings for old names
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# 1. New names exist and are importable
# ---------------------------------------------------------------------------


class TestNewNamesExist:
    """New class names must be importable from the public API."""

    def test_classical_krylov_diag_importable(self):
        from qvartools.krylov.basis.skqd import ClassicalKrylovDiagonalization

        assert ClassicalKrylovDiagonalization is not None

    def test_flow_guided_krylov_diag_importable(self):
        from qvartools.krylov.basis.flow_guided import FlowGuidedKrylovDiag

        assert FlowGuidedKrylovDiag is not None

    def test_new_names_in_krylov_init(self):
        from qvartools.krylov import (
            ClassicalKrylovDiagonalization,
            FlowGuidedKrylovDiag,
        )

        assert ClassicalKrylovDiagonalization is not None
        assert FlowGuidedKrylovDiag is not None

    def test_skqd_config_unchanged(self):
        from qvartools.krylov import SKQDConfig

        cfg = SKQDConfig(max_krylov_dim=3)
        assert cfg.max_krylov_dim == 3


# ---------------------------------------------------------------------------
# 2. Deprecated aliases still work but warn
# ---------------------------------------------------------------------------


class TestDeprecatedAliases:
    """Old names must still resolve but emit DeprecationWarning."""

    def test_sample_based_krylov_alias_warns(self):
        import warnings

        from qvartools.krylov.basis.skqd import ClassicalKrylovDiagonalization

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from qvartools.krylov.basis import skqd

            cls = getattr(skqd, "SampleBasedKrylovDiagonalization")
            assert cls is ClassicalKrylovDiagonalization
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_flow_guided_skqd_alias_warns(self):
        import warnings

        from qvartools.krylov.basis.flow_guided import FlowGuidedKrylovDiag

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from qvartools.krylov.basis import flow_guided

            cls = getattr(flow_guided, "FlowGuidedSKQD")
            assert cls is FlowGuidedKrylovDiag
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()


# ---------------------------------------------------------------------------
# 3. Classical Krylov works with new name
# ---------------------------------------------------------------------------


class TestClassicalKrylovDiag:
    """ClassicalKrylovDiagonalization must function identically to old name."""

    def test_runs_and_returns_energy(self, heisenberg_4):
        import numpy as np

        from qvartools.krylov.basis.skqd import (
            ClassicalKrylovDiagonalization,
            SKQDConfig,
        )

        config = SKQDConfig(max_krylov_dim=3, time_step=0.1, shots_per_krylov=200)
        solver = ClassicalKrylovDiagonalization(heisenberg_4, config)
        eigenvalues, info = solver.run()
        assert np.isfinite(eigenvalues[0])
        assert info["krylov_dim"] == 3


class TestFlowGuidedKrylovDiag:
    """FlowGuidedKrylovDiag must function identically to old name."""

    def test_runs_with_nf_basis(self, heisenberg_4):
        import numpy as np

        from qvartools.krylov.basis.flow_guided import FlowGuidedKrylovDiag
        from qvartools.krylov.basis.skqd import SKQDConfig

        config = SKQDConfig(max_krylov_dim=2, time_step=0.1, shots_per_krylov=200)
        nf_basis = torch.randint(0, 2, (8, 4), dtype=torch.int64)
        solver = FlowGuidedKrylovDiag(heisenberg_4, config, nf_basis=nf_basis)
        results = solver.run_with_nf()
        assert np.isfinite(results["energy"])


# ---------------------------------------------------------------------------
# 4. Pipeline routing
# ---------------------------------------------------------------------------


class TestPipelineRouting:
    """Pipeline subspace_mode routing must match ADR-001."""

    def test_classical_krylov_mode_exists(self):
        from qvartools.pipeline_config import PipelineConfig

        cfg = PipelineConfig(subspace_mode="classical_krylov")
        assert cfg.subspace_mode == "classical_krylov"

    def test_default_mode_is_classical_krylov(self):
        """Default subspace_mode should be 'classical_krylov' for safety."""
        from qvartools.pipeline_config import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.subspace_mode == "classical_krylov"

    def test_pipeline_routes_classical_krylov(self, heisenberg_4):
        """subspace_mode='classical_krylov' should use ClassicalKrylovDiagonalization."""
        from qvartools.pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            use_particle_conserving_flow=False,
            nf_hidden_dims=[16, 8],
            nqs_hidden_dims=[16, 8],
            samples_per_batch=20,
            num_batches=1,
            max_epochs=2,
            min_epochs=1,
            max_accumulated_basis=20,
            use_diversity_selection=False,
            use_residual_expansion=False,
            max_krylov_dim=2,
            shots_per_krylov=50,
            subspace_mode="classical_krylov",
            device="cpu",
        )

        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=heisenberg_4, config=config, auto_adapt=False
        )
        results = pipeline.run(progress=False)
        assert "final_energy" in results
        import numpy as np

        assert np.isfinite(results["final_energy"])

    def test_skqd_mode_routes_to_quantum(self):
        """subspace_mode='skqd' should route to _run_skqd_quantum (not classical)."""
        from unittest.mock import MagicMock

        from qvartools.pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(subspace_mode="skqd")
        pipeline = MagicMock(spec=FlowGuidedKrylovPipeline)
        pipeline.config = config
        pipeline.nf_basis = torch.ones(2, 4)

        # Call the real routing method on the mock
        FlowGuidedKrylovPipeline.run_subspace_diag(pipeline)
        pipeline._run_skqd_quantum.assert_called_once()
        pipeline._run_classical_krylov.assert_not_called()

    def test_unknown_mode_warns_and_falls_back(self, heisenberg_4, caplog):
        """Unknown subspace_mode should log a warning."""
        import logging

        from qvartools.pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            subspace_mode="typo_mode",
            use_particle_conserving_flow=False,
            nf_hidden_dims=[16],
            nqs_hidden_dims=[16],
            samples_per_batch=10,
            num_batches=1,
            max_epochs=1,
            min_epochs=1,
            skip_skqd=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=heisenberg_4, config=config, auto_adapt=False
        )
        pipeline.train_flow_nqs(progress=False)
        pipeline.extract_and_select_basis()

        # Enable propagation on root qvartools logger so caplog can capture
        root_qv = logging.getLogger("qvartools")
        old_propagate = root_qv.propagate
        root_qv.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="qvartools"):
                pipeline.run_subspace_diag(progress=False)
            assert "Unknown subspace_mode" in caplog.text
        finally:
            root_qv.propagate = old_propagate
