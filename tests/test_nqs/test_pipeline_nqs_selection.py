"""Tests for NQS type selection in pipeline (ADR-001 Phase 2.3).

Verifies that PipelineConfig.nqs_type controls which NQS is created,
and that adapters are wired correctly into the pipeline.
"""

from __future__ import annotations


class TestNQSTypeConfig:
    """PipelineConfig must support nqs_type parameter."""

    def test_default_nqs_type_is_dense(self):
        from qvartools.pipeline_config import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.nqs_type == "dense"

    def test_all_valid_nqs_types(self):
        from qvartools.pipeline_config import PipelineConfig

        for nqs_type in ("dense", "signed", "complex", "rbm", "transformer"):
            cfg = PipelineConfig(nqs_type=nqs_type)
            assert cfg.nqs_type == nqs_type


class TestPipelineNQSSelection:
    """Pipeline must instantiate the correct NQS based on config."""

    def test_dense_nqs_created_by_default(self, heisenberg_4):
        from qvartools.nqs import DenseNQS
        from qvartools.pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            use_particle_conserving_flow=False,
            nf_hidden_dims=[16],
            nqs_hidden_dims=[16],
            samples_per_batch=10,
            num_batches=1,
            max_epochs=1,
            min_epochs=1,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=heisenberg_4, config=config, auto_adapt=False
        )
        assert isinstance(pipeline.nqs, DenseNQS)

    def test_complex_nqs_created(self, heisenberg_4):
        from qvartools.nqs import ComplexNQS
        from qvartools.pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            nqs_type="complex",
            use_particle_conserving_flow=False,
            nf_hidden_dims=[16],
            nqs_hidden_dims=[16],
            samples_per_batch=10,
            num_batches=1,
            max_epochs=1,
            min_epochs=1,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=heisenberg_4, config=config, auto_adapt=False
        )
        assert isinstance(pipeline.nqs, ComplexNQS)

    def test_signed_nqs_created(self, heisenberg_4):
        from qvartools.nqs import SignedDenseNQS
        from qvartools.pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            nqs_type="signed",
            use_particle_conserving_flow=False,
            nf_hidden_dims=[16],
            nqs_hidden_dims=[16],
            samples_per_batch=10,
            num_batches=1,
            max_epochs=1,
            min_epochs=1,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=heisenberg_4, config=config, auto_adapt=False
        )
        assert isinstance(pipeline.nqs, SignedDenseNQS)

    def test_rbm_nqs_created(self, heisenberg_4):
        from qvartools.nqs import RBMQuantumState
        from qvartools.pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        config = PipelineConfig(
            nqs_type="rbm",
            use_particle_conserving_flow=False,
            nf_hidden_dims=[16],
            nqs_hidden_dims=[16],
            samples_per_batch=10,
            num_batches=1,
            max_epochs=1,
            min_epochs=1,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=heisenberg_4, config=config, auto_adapt=False
        )
        assert isinstance(pipeline.nqs, RBMQuantumState)
