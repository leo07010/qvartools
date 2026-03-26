"""Regression test for TransformerNFSampler._build_nqs bug (ADR-001)."""

from __future__ import annotations

import torch.nn as nn


class TestBuildNQSFallback:
    """_build_nqs must not crash when falling back to DenseNQS."""

    def test_build_nqs_returns_module(self):
        from qvartools.samplers.classical.transformer_nf_sampler import (
            TransformerNFSampler,
        )

        nqs = TransformerNFSampler._build_nqs(
            n_sites=4, embed_dim=16, n_heads=2, n_layers=1
        )
        assert isinstance(nqs, nn.Module)

    def test_build_nqs_has_log_amplitude(self):
        from qvartools.samplers.classical.transformer_nf_sampler import (
            TransformerNFSampler,
        )

        nqs = TransformerNFSampler._build_nqs(
            n_sites=4, embed_dim=16, n_heads=2, n_layers=1
        )
        assert hasattr(nqs, "log_amplitude")
