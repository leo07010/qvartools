"""Integration test: full pipeline on H2 molecule (requires PySCF)."""

from __future__ import annotations

import numpy as np
import pytest

from qvartools.pipeline import FlowGuidedKrylovPipeline, PipelineConfig, run_molecular_benchmark

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
def h2_hamiltonian():
    """H2 molecule at 0.74 angstrom, sto-3g basis (4 qubits)."""
    if not _HAS_PYSCF:
        pytest.skip("PySCF is not installed")
    from qvartools.hamiltonians import MolecularHamiltonian, compute_molecular_integrals

    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
    integrals = compute_molecular_integrals(geometry, basis="sto-3g")
    return integrals, MolecularHamiltonian(integrals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestH2Pipeline:
    """Integration tests for H2 molecule pipeline."""

    @skip_no_pyscf
    def test_full_pipeline_h2(self, h2_hamiltonian):
        """Run full FlowGuidedKrylovPipeline on H2 with minimal config."""
        integrals, hamiltonian = h2_hamiltonian

        config = PipelineConfig(
            use_particle_conserving_flow=True,
            nf_hidden_dims=[32, 16],
            nqs_hidden_dims=[32, 16],
            samples_per_batch=50,
            num_batches=2,
            max_epochs=10,
            min_epochs=2,
            max_accumulated_basis=50,
            use_diversity_selection=False,
            use_residual_expansion=False,
            max_krylov_dim=2,
            time_step=0.1,
            shots_per_krylov=200,
            skip_skqd=False,
            device="cpu",
        )

        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=hamiltonian,
            config=config,
            auto_adapt=False,
        )
        results = pipeline.run(progress=False)

        final_energy = results["final_energy"]
        assert np.isfinite(final_energy)
        # H2 should have a bound state (negative energy)
        assert final_energy < 0.0

    @skip_no_pyscf
    def test_h2_energy_reasonable(self, h2_hamiltonian):
        """H2 energy should be better than or close to HF energy (~-1.117)."""
        _, hamiltonian = h2_hamiltonian

        config = PipelineConfig(
            use_particle_conserving_flow=True,
            nf_hidden_dims=[32, 16],
            nqs_hidden_dims=[32, 16],
            samples_per_batch=50,
            num_batches=2,
            max_epochs=10,
            min_epochs=2,
            max_accumulated_basis=50,
            use_diversity_selection=False,
            use_residual_expansion=False,
            max_krylov_dim=3,
            shots_per_krylov=500,
            skip_skqd=False,
            device="cpu",
        )

        pipeline = FlowGuidedKrylovPipeline(
            hamiltonian=hamiltonian,
            config=config,
            auto_adapt=False,
        )
        results = pipeline.run(progress=False)

        # HF energy for H2/sto-3g is approximately -1.117
        # Our energy should be at least in the right ballpark
        assert results["final_energy"] < -0.5

    @skip_no_pyscf
    def test_run_molecular_benchmark_h2(self):
        """Test the convenience run_molecular_benchmark function."""
        config = PipelineConfig(
            nf_hidden_dims=[32, 16],
            nqs_hidden_dims=[32, 16],
            samples_per_batch=50,
            num_batches=2,
            max_epochs=5,
            min_epochs=1,
            max_accumulated_basis=30,
            use_diversity_selection=False,
            use_residual_expansion=False,
            max_krylov_dim=2,
            shots_per_krylov=100,
            skip_skqd=False,
            device="cpu",
        )
        results = run_molecular_benchmark("h2", config=config, verbose=False)
        assert np.isfinite(results["final_energy"])
        assert results["final_energy"] < 0.0
