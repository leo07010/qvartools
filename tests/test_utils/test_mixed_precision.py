"""Tests for mixed-precision eigendecomposition.

Tests for mixed_precision_eigh: FP32 solve + FP64 Rayleigh quotient
refinement for GPU-accelerated eigenvalue computation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestMixedPrecisionEigh:
    """Tests for mixed_precision_eigh."""

    def test_import(self) -> None:
        """mixed_precision_eigh should be importable."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        assert callable(mixed_precision_eigh)

    def test_correctness_matches_fp64(self) -> None:
        """Mixed-precision result should match FP64 reference within 1e-6."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        rng = np.random.default_rng(42)
        A = rng.standard_normal((200, 200))
        H = A + A.T  # symmetric
        H_torch = torch.tensor(H, dtype=torch.float64)

        eigenvalues, eigenvectors = mixed_precision_eigh(H_torch)

        ref_vals = np.sort(np.linalg.eigvalsh(H))
        np.testing.assert_allclose(eigenvalues.numpy(), ref_vals, atol=1e-6, rtol=1e-6)

    def test_returns_sorted_eigenvalues(self) -> None:
        """Eigenvalues should be sorted ascending."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        H = torch.diag(torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0], dtype=torch.float64))
        vals, vecs = mixed_precision_eigh(H)
        assert torch.all(vals[:-1] <= vals[1:])
        torch.testing.assert_close(
            vals, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        )

    def test_small_matrix_skips_mixed_precision(self) -> None:
        """For n <= 64, should use direct FP64 (no FP32 step)."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        H = torch.diag(torch.tensor([3.0, 1.0, 2.0], dtype=torch.float64))
        vals, vecs = mixed_precision_eigh(H)
        torch.testing.assert_close(
            vals, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        )

    def test_already_fp64_input(self) -> None:
        """FP64 input should work and produce correct results."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        H = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
        vals, _ = mixed_precision_eigh(H)
        torch.testing.assert_close(vals, torch.tensor([1.0, 2.0], dtype=torch.float64))

    def test_fp32_input_accepted(self) -> None:
        """FP32 input should be accepted and produce correct results."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        H = torch.diag(torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32))
        vals, _ = mixed_precision_eigh(H)
        assert vals.dtype == torch.float64  # output always FP64
        np.testing.assert_allclose(vals.numpy(), [1.0, 2.0, 3.0], atol=1e-5)

    def test_output_always_fp64(self) -> None:
        """Output eigenvalues and eigenvectors should always be FP64."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        H = torch.eye(4, dtype=torch.float32)
        vals, vecs = mixed_precision_eigh(H)
        assert vals.dtype == torch.float64
        assert vecs.dtype == torch.float64

    def test_eigenvectors_orthonormal(self) -> None:
        """Eigenvectors should be orthonormal."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        rng = np.random.default_rng(99)
        A = rng.standard_normal((100, 100))
        H = torch.tensor(A + A.T, dtype=torch.float64)
        _, vecs = mixed_precision_eigh(H)
        eye = vecs.T @ vecs
        torch.testing.assert_close(
            eye, torch.eye(100, dtype=torch.float64), atol=1e-6, rtol=0
        )

    @pytest.mark.gpu
    def test_gpu_acceleration(self) -> None:
        """On GPU, should produce correct results."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        H = torch.diag(torch.tensor([5.0, 1.0, 3.0], dtype=torch.float64)).cuda()
        vals, vecs = mixed_precision_eigh(H)
        # Results should be on CPU (consistent with gpu_eigh convention)
        assert vals.device == torch.device("cpu")
        torch.testing.assert_close(
            vals, torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
        )

    @pytest.mark.gpu
    def test_gpu_large_matrix_correctness(self) -> None:
        """Large matrix on GPU should match CPU reference."""
        from qvartools._utils.gpu.linear_algebra import mixed_precision_eigh

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        rng = np.random.default_rng(42)
        A = rng.standard_normal((500, 500))
        H = torch.tensor(A + A.T, dtype=torch.float64).cuda()
        vals, _ = mixed_precision_eigh(H)
        ref = np.sort(np.linalg.eigvalsh(A + A.T))
        np.testing.assert_allclose(vals.numpy(), ref, atol=1e-5)
