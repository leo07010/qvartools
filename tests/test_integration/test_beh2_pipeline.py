"""Integration test: full pipeline on BeH2 molecule (requires PySCF)."""

from __future__ import annotations

import numpy as np
import pytest

from qvartools.pipeline import FlowGuidedKrylovPipeline, PipelineConfig

# Mark entire module as requiring PySCF
pytestmark = pytest.mark.pyscf

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = True
except ImportError:
    _HAS_PYSCF = False

skip_no_pyscf = pytest.mark.skipif(not _HAS_PYSCF, reason="PySCF is not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def beh2_hamiltonian():
    """BeH2 molecule, sto-6g basis (14 qubits)."""
    if not _HAS_PYSCF:
        pytest.skip("PySCF is not installed")
    from qvartools.hamiltonians import MolecularHamiltonian, compute_molecular_integrals

    geometry = [
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.3264)),
        ("H", (0.0, 0.0, -1.3264)),
    ]
    integrals = compute_molecular_integrals(geometry, basis="sto-6g")
    return MolecularHamiltonian(integrals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBeH2Pipeline:
    """Integration tests for BeH2 molecule pipeline."""

    @skip_no_pyscf
    def test_full_pipeline_beh2(self, beh2_hamiltonian):
        """Run full pipeline on BeH2 with small config (fast)."""
        config = PipelineConfig(
            use_particle_conserving_flow=True,
            nf_hidden_dims=[64, 32],
            nqs_hidden_dims=[64, 32],
            samples_per_batch=100,
            num_batches=2,
            max_epochs=10,
            min_epochs=2,
            max_accumulated_basis=100,
            use_diversity_selection=False,
            use_residual_expansion=False,
            max_krylov_dim=2,
            time_step=0.1,
            shots_per_krylov=200,
            skip_skqd=False,
            device="cpu",
        )

        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=beh2_hamiltonian,
            config=config,
            auto_adapt=False,
        )
        results = pipeline.run(progress=False)

        final_energy = results["final_energy"]
        assert np.isfinite(final_energy)
        # BeH2 should have a negative energy (bound state)
        assert final_energy < 0.0

    @skip_no_pyscf
    def test_beh2_14_qubits(self, beh2_hamiltonian):
        """Verify BeH2 uses 14 qubits and pipeline handles multi-atom system."""
        assert beh2_hamiltonian.num_sites == 14

        config = PipelineConfig(
            use_particle_conserving_flow=True,
            nf_hidden_dims=[32, 16],
            nqs_hidden_dims=[32, 16],
            samples_per_batch=50,
            num_batches=2,
            max_epochs=5,
            min_epochs=1,
            max_accumulated_basis=50,
            use_diversity_selection=False,
            use_residual_expansion=False,
            skip_skqd=True,  # skip SKQD for speed on larger system
            device="cpu",
        )

        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=beh2_hamiltonian,
            config=config,
            auto_adapt=False,
        )
        results = pipeline.run(progress=False)
        assert np.isfinite(results["final_energy"])
