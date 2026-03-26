"""Tests for persistent Hamiltonian integral caching via joblib."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("joblib")
pytest.importorskip("pyscf")


class TestHamiltonianCache:
    """Tests for cached_compute_molecular_integrals."""

    def test_import(self) -> None:
        """cached_compute_molecular_integrals should be importable."""
        from qvartools.hamiltonians.integrals import cached_compute_molecular_integrals

        assert callable(cached_compute_molecular_integrals)

    def test_returns_molecular_integrals(self, tmp_path) -> None:
        """Should return a MolecularIntegrals dataclass."""
        from qvartools.hamiltonians.integrals import (
            MolecularIntegrals,
            get_integral_cache,
        )

        cache = get_integral_cache(str(tmp_path / "cache"))
        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
        result = cache(geometry, basis="sto-3g")
        assert isinstance(result, MolecularIntegrals)
        assert result.n_orbitals == 2
        assert result.n_electrons == 2

    def test_second_call_is_cached(self, tmp_path) -> None:
        """Second call with same args should reuse on-disk cache."""
        from qvartools.hamiltonians.integrals import get_integral_cache

        cache_dir = tmp_path / "cache"
        cache = get_integral_cache(str(cache_dir))
        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]

        # First call — should create cache entries on disk
        r1 = cache(geometry, basis="sto-3g")
        cache_files_after_first = sorted(
            p.relative_to(cache_dir) for p in cache_dir.rglob("*") if p.is_file()
        )
        assert len(cache_files_after_first) > 0

        # Second call — should reuse existing cache (no new files)
        r2 = cache(geometry, basis="sto-3g")
        cache_files_after_second = sorted(
            p.relative_to(cache_dir) for p in cache_dir.rglob("*") if p.is_file()
        )
        assert cache_files_after_second == cache_files_after_first

        # Results should match
        np.testing.assert_array_equal(r1.h1e, r2.h1e)
        np.testing.assert_array_equal(r1.h2e, r2.h2e)
        assert r1.nuclear_repulsion == r2.nuclear_repulsion

    def test_different_geometry_not_cached(self, tmp_path) -> None:
        """Different geometry should compute fresh result."""
        from qvartools.hamiltonians.integrals import get_integral_cache

        cache = get_integral_cache(str(tmp_path / "cache"))
        g1 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
        g2 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))]

        r1 = cache(g1, basis="sto-3g")
        r2 = cache(g2, basis="sto-3g")

        # Different geometry → different integrals
        assert not np.allclose(r1.h1e, r2.h1e)

    def test_clear_cache(self, tmp_path) -> None:
        """clear_integral_cache should remove cached data."""
        from qvartools.hamiltonians.integrals import (
            clear_integral_cache,
            get_integral_cache,
        )

        cache_dir = str(tmp_path / "qvartools_cache")
        cache = get_integral_cache(cache_dir)
        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
        cache(geometry, basis="sto-3g")

        clear_integral_cache(cache_dir)

        cache_path = pathlib.Path(cache_dir)
        if cache_path.exists():
            remaining = [p for p in cache_path.rglob("*") if p.is_file()]
            assert len(remaining) == 0
