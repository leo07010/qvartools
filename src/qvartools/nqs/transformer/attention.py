"""
attention --- Attention mechanisms for transformer NQS
======================================================

Provides :class:`CausalSelfAttention` and :class:`CrossAttention`, the
core attention building blocks used by the autoregressive transformer
neural quantum state architecture.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CausalSelfAttention",
    "CrossAttention",
]


# ---------------------------------------------------------------------------
# CausalSelfAttention
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional KV cache.

    Applies a causal (lower-triangular) attention mask so that each
    position can only attend to itself and preceding positions.  An
    optional KV cache enables efficient autoregressive generation.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    n_heads : int
        Number of attention heads.  ``embed_dim`` must be divisible by
        ``n_heads``.
    dropout : float, optional
        Dropout probability on attention weights (default ``0.0``).

    Attributes
    ----------
    qkv_proj : nn.Linear
        Joint projection for queries, keys, and values.
    out_proj : nn.Linear
        Output projection after multi-head attention.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})."
            )
        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.head_dim: int = embed_dim // n_heads
        self.dropout: float = dropout

        self.qkv_proj: nn.Linear = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)

        # KV cache state
        self._cache_enabled: bool = False
        self._k_cache: torch.Tensor | None = None
        self._v_cache: torch.Tensor | None = None

    def enable_cache(self) -> None:
        """Enable KV cache for autoregressive generation."""
        self._cache_enabled = True
        self.clear_cache()

    def disable_cache(self) -> None:
        """Disable KV cache and free cached tensors."""
        self._cache_enabled = False
        self.clear_cache()

    def clear_cache(self) -> None:
        """Clear the stored KV cache tensors."""
        self._k_cache = None
        self._v_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor, shape ``(batch, seq_len, embed_dim)``.
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        # (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # KV cache handling
        if self._cache_enabled:
            if self._k_cache is not None:
                k = torch.cat([self._k_cache, k], dim=2)  # type: ignore[list-item]
                v = torch.cat([self._v_cache, v], dim=2)  # type: ignore[list-item]
            self._k_cache = k.detach()
            self._v_cache = v.detach()

        # Scaled dot-product attention with causal mask
        kv_len = k.shape[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
            diagonal=kv_len - seq_len + 1,
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        # Weighted sum of values
        attn_output = torch.matmul(
            attn_weights, v
        )  # (batch, n_heads, seq_len, head_dim)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------


class CrossAttention(nn.Module):
    """Multi-head cross-attention for alpha-to-beta coupling.

    The query comes from one sequence (e.g. beta orbitals) while the
    keys and values come from another (e.g. alpha orbitals).

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embeddings.
    n_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability on attention weights (default ``0.0``).

    Attributes
    ----------
    q_proj : nn.Linear
        Query projection.
    kv_proj : nn.Linear
        Joint key-value projection.
    out_proj : nn.Linear
        Output projection.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})."
            )
        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.head_dim: int = embed_dim // n_heads
        self.dropout: float = dropout

        self.q_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.kv_proj: nn.Linear = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention from query to key_value.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor from the target sequence (e.g. beta),
            shape ``(batch, q_len, embed_dim)``.
        key_value : torch.Tensor
            Key/value tensor from the source sequence (e.g. alpha),
            shape ``(batch, kv_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Cross-attended output, shape ``(batch, q_len, embed_dim)``.
        """
        batch, q_len, _ = query.shape
        kv_len = key_value.shape[1]

        q = self.q_proj(query)
        kv = self.kv_proj(key_value)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head
        q = q.view(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (no causal mask for cross-attention)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch, q_len, self.embed_dim)
        )

        return self.out_proj(attn_output)
