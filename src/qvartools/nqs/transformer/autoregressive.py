"""
transformer --- Autoregressive transformer NQS with KV cache
=============================================================

Provides :class:`TransformerBlock` and :class:`AutoregressiveTransformer`,
an autoregressive transformer architecture for neural quantum states with
separate alpha and beta spin channels.  The beta channel cross-attends to
the alpha channel, enabling spin-spin correlations.

Key features:

* Pre-norm transformer blocks with optional cross-attention.
* Particle-conserving autoregressive sampling (enforces exact electron
  counts in each spin channel).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qvartools.nqs.transformer.attention import CausalSelfAttention, CrossAttention

__all__ = [
    "TransformerBlock",
    "AutoregressiveTransformer",
]


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional cross-attention.

    Architecture (pre-norm style):

    1. LayerNorm -> CausalSelfAttention -> residual
    2. (Optional) LayerNorm -> CrossAttention -> residual
    3. LayerNorm -> FFN (Linear -> GELU -> Linear) -> residual

    Parameters
    ----------
    embed_dim : int
        Embedding dimensionality.
    n_heads : int
        Number of attention heads.
    ffn_dim : int
        Hidden dimensionality of the feed-forward network.
    dropout : float, optional
        Dropout probability (default ``0.0``).
    has_cross_attn : bool, optional
        Whether to include a cross-attention sub-layer
        (default ``False``).

    Attributes
    ----------
    self_attn : CausalSelfAttention
        Causal self-attention layer.
    cross_attn : CrossAttention or None
        Cross-attention layer (only if ``has_cross_attn``).
    ffn : nn.Sequential
        Two-layer feed-forward network with GELU activation.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        has_cross_attn: bool = False,
    ) -> None:
        super().__init__()

        # Self-attention
        self.ln_sa: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.self_attn: CausalSelfAttention = CausalSelfAttention(
            embed_dim, n_heads, dropout=dropout
        )

        # Cross-attention (optional)
        self.cross_attn: CrossAttention | None = None
        self.ln_ca: nn.LayerNorm | None = None
        if has_cross_attn:
            self.ln_ca = nn.LayerNorm(embed_dim)
            self.cross_attn = CrossAttention(embed_dim, n_heads, dropout=dropout)

        # Feed-forward network
        self.ln_ffn: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cross_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``(batch, seq_len, embed_dim)``.
        cross_kv : torch.Tensor or None, optional
            Key/value tensor for cross-attention, shape
            ``(batch, kv_len, embed_dim)``.  Required when
            ``has_cross_attn`` is ``True``.

        Returns
        -------
        torch.Tensor
            Output tensor, shape ``(batch, seq_len, embed_dim)``.
        """
        # Self-attention with residual
        x = x + self.self_attn(self.ln_sa(x))

        # Cross-attention with residual (optional)
        if self.cross_attn is not None and cross_kv is not None:
            assert self.ln_ca is not None
            x = x + self.cross_attn(self.ln_ca(x), cross_kv)

        # FFN with residual
        x = x + self.ffn(self.ln_ffn(x))

        return x


# ---------------------------------------------------------------------------
# AutoregressiveTransformer
# ---------------------------------------------------------------------------


class AutoregressiveTransformer(nn.Module):
    """Autoregressive transformer NQS with alpha/beta spin channels.

    Models the joint probability of occupying orbitals by factorising
    it autoregressively:

    .. math::

        p(\\mathbf{x}) = \\prod_{i=1}^{N_{\\text{orb}}}
        p(x^\\alpha_i | x^\\alpha_{<i})
        \\;\\prod_{i=1}^{N_{\\text{orb}}}
        p(x^\\beta_i | x^\\beta_{<i}, \\mathbf{x}^\\alpha)

    The alpha channel uses causal self-attention only.  The beta channel
    uses causal self-attention *plus* cross-attention to the full alpha
    representation, enabling spin-spin correlations.

    Sampling enforces particle conservation: exactly ``n_alpha``
    electrons in the alpha channel and ``n_beta`` in the beta channel.

    Parameters
    ----------
    n_orbitals : int
        Number of spatial orbitals per spin channel.
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.
    embed_dim : int, optional
        Embedding dimensionality (default ``64``).
    n_heads : int, optional
        Number of attention heads (default ``4``).
    n_layers : int, optional
        Number of transformer layers per channel (default ``4``).
    dropout : float, optional
        Dropout probability (default ``0.0``).

    Attributes
    ----------
    alpha_blocks : nn.ModuleList
        Transformer blocks for the alpha channel (self-attention only).
    beta_blocks : nn.ModuleList
        Transformer blocks for the beta channel (self + cross-attention).
    alpha_head : nn.Linear
        Output head producing alpha occupation logits.
    beta_head : nn.Linear
        Output head producing beta occupation logits.

    Examples
    --------
    >>> model = AutoregressiveTransformer(
    ...     n_orbitals=6, n_alpha=2, n_beta=2, embed_dim=32, n_heads=4
    ... )
    >>> configs = model.sample(n_samples=16)  # shape (16, 12)
    """

    def __init__(
        self,
        n_orbitals: int,
        n_alpha: int,
        n_beta: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_alpha > n_orbitals:
            raise ValueError(
                f"n_alpha ({n_alpha}) cannot exceed n_orbitals ({n_orbitals})."
            )
        if n_beta > n_orbitals:
            raise ValueError(
                f"n_beta ({n_beta}) cannot exceed n_orbitals ({n_orbitals})."
            )

        self.n_orbitals: int = n_orbitals
        self.n_alpha: int = n_alpha
        self.n_beta: int = n_beta
        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers

        ffn_dim = 4 * embed_dim

        # Token embedding: occupation {0, 1} -> embed_dim
        # We use 3 tokens: 0 (unoccupied), 1 (occupied), 2 (start token)
        self.token_embed: nn.Embedding = nn.Embedding(3, embed_dim)

        # Positional embedding for each orbital position
        self.pos_embed_alpha: nn.Parameter = nn.Parameter(
            torch.randn(1, n_orbitals, embed_dim) * 0.02
        )
        self.pos_embed_beta: nn.Parameter = nn.Parameter(
            torch.randn(1, n_orbitals, embed_dim) * 0.02
        )

        # Alpha transformer blocks (self-attention only)
        self.alpha_blocks: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    n_heads,
                    ffn_dim,
                    dropout=dropout,
                    has_cross_attn=False,
                )
                for _ in range(n_layers)
            ]
        )

        # Beta transformer blocks (self-attention + cross-attention to alpha)
        self.beta_blocks: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    n_heads,
                    ffn_dim,
                    dropout=dropout,
                    has_cross_attn=True,
                )
                for _ in range(n_layers)
            ]
        )

        # Output heads
        self.ln_alpha: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.alpha_head: nn.Linear = nn.Linear(embed_dim, 1)

        self.ln_beta: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.beta_head: nn.Linear = nn.Linear(embed_dim, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_sequence(
        self,
        tokens: torch.Tensor,
        pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a token sequence with positional encoding.

        Parameters
        ----------
        tokens : torch.Tensor
            Integer token indices, shape ``(batch, seq_len)``.
        pos_embed : torch.Tensor
            Positional embeddings, shape ``(1, max_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Embedded sequence, shape ``(batch, seq_len, embed_dim)``.
        """
        seq_len = tokens.shape[1]
        tok_emb = self.token_embed(tokens)  # (batch, seq_len, embed_dim)
        return tok_emb + pos_embed[:, :seq_len, :]

    def _run_alpha(self, alpha_tokens: torch.Tensor) -> torch.Tensor:
        """Run the alpha transformer stack.

        Parameters
        ----------
        alpha_tokens : torch.Tensor
            Alpha token sequence, shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Alpha representations, shape ``(batch, seq_len, embed_dim)``.
        """
        h = self._embed_sequence(alpha_tokens, self.pos_embed_alpha)
        for block in self.alpha_blocks:
            h = block(h)
        return h

    def _run_beta(
        self,
        beta_tokens: torch.Tensor,
        alpha_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Run the beta transformer stack with cross-attention to alpha.

        Parameters
        ----------
        beta_tokens : torch.Tensor
            Beta token sequence, shape ``(batch, seq_len)``.
        alpha_repr : torch.Tensor
            Alpha representations for cross-attention,
            shape ``(batch, n_orbitals, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Beta representations, shape ``(batch, seq_len, embed_dim)``.
        """
        h = self._embed_sequence(beta_tokens, self.pos_embed_beta)
        for block in self.beta_blocks:
            h = block(h, cross_kv=alpha_repr)
        return h

    def _enable_cache(self) -> None:
        """Enable KV cache in all causal self-attention layers."""
        for block in self.alpha_blocks:
            block.self_attn.enable_cache()  # type: ignore[union-attr,operator]
        for block in self.beta_blocks:
            block.self_attn.enable_cache()  # type: ignore[union-attr,operator]

    def _disable_cache(self) -> None:
        """Disable KV cache in all causal self-attention layers."""
        for block in self.alpha_blocks:
            block.self_attn.disable_cache()  # type: ignore[union-attr,operator]
        for block in self.beta_blocks:
            block.self_attn.disable_cache()  # type: ignore[union-attr,operator]

    def _clear_cache(self) -> None:
        """Clear KV cache in all causal self-attention layers."""
        for block in self.alpha_blocks:
            block.self_attn.clear_cache()  # type: ignore[union-attr,operator]
        for block in self.beta_blocks:
            block.self_attn.clear_cache()  # type: ignore[union-attr,operator]

    # ------------------------------------------------------------------
    # Log probability
    # ------------------------------------------------------------------

    def log_prob(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log-probability of a configuration.

        Parameters
        ----------
        alpha : torch.Tensor
            Alpha spin-orbital occupations, shape ``(batch, n_orbitals)``
            with entries in ``{0, 1}``.
        beta : torch.Tensor
            Beta spin-orbital occupations, shape ``(batch, n_orbitals)``
            with entries in ``{0, 1}``.

        Returns
        -------
        torch.Tensor
            Log-probabilities, shape ``(batch,)``.
        """
        batch = alpha.shape[0]
        device = alpha.device

        # Build alpha input: [start_token, x_1, ..., x_{N-1}]
        # (teacher forcing: shift right, prepend start token)
        start_tok = torch.full(
            (batch, 1), fill_value=2, dtype=torch.long, device=device
        )
        alpha_input = torch.cat([start_tok, alpha[:, :-1].long()], dim=1)

        # Alpha forward
        alpha_repr = self._run_alpha(alpha_input)
        alpha_logits = self.alpha_head(self.ln_alpha(alpha_repr)).squeeze(-1)
        # (batch, n_orbitals)

        # Alpha log-prob via binary cross-entropy
        alpha_log_probs = -F.binary_cross_entropy_with_logits(
            alpha_logits, alpha.float(), reduction="none"
        )  # (batch, n_orbitals)

        # Build beta input: [start_token, x_1, ..., x_{N-1}]
        beta_input = torch.cat([start_tok, beta[:, :-1].long()], dim=1)

        # Beta forward with cross-attention to alpha
        beta_repr = self._run_beta(beta_input, alpha_repr)
        beta_logits = self.beta_head(self.ln_beta(beta_repr)).squeeze(-1)

        beta_log_probs = -F.binary_cross_entropy_with_logits(
            beta_logits, beta.float(), reduction="none"
        )  # (batch, n_orbitals)

        # Total log-prob = sum over all orbital positions
        return alpha_log_probs.sum(dim=-1) + beta_log_probs.sum(dim=-1)

    # ------------------------------------------------------------------
    # Autoregressive sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate particle-conserving configurations autoregressively.

        Samples alpha orbitals first (enforcing exactly ``n_alpha``
        electrons), then samples beta orbitals with cross-attention to
        alpha (enforcing exactly ``n_beta`` electrons).  The returned
        configuration is ``[alpha, beta]`` concatenated along the orbital
        axis.

        KV caching is used for efficient autoregressive generation.

        Parameters
        ----------
        n_samples : int
            Number of configurations to generate.
        temperature : float, optional
            Sampling temperature.  Values > 1 increase randomness;
            values < 1 sharpen the distribution (default ``1.0``).

        Returns
        -------
        torch.Tensor
            Sampled configurations, shape ``(n_samples, 2 * n_orbitals)``
            with entries in ``{0, 1}``.  The first ``n_orbitals`` columns
            are alpha occupations and the last ``n_orbitals`` are beta.
        """
        device = next(self.parameters()).device
        n_orb = self.n_orbitals

        alpha_config = torch.zeros(n_samples, n_orb, dtype=torch.long, device=device)
        beta_config = torch.zeros(n_samples, n_orb, dtype=torch.long, device=device)

        # --- Sample alpha channel ---
        self._enable_cache()
        try:
            alpha_config = self._sample_channel(
                alpha_config,
                n_electrons=self.n_alpha,
                pos_embed=self.pos_embed_alpha,
                blocks=self.alpha_blocks,
                head=self.alpha_head,
                ln=self.ln_alpha,
                temperature=temperature,
                cross_kv=None,
            )

            # Get full alpha representation for beta cross-attention
            self._clear_cache()
            start_tok = torch.full(
                (n_samples, 1), fill_value=2, dtype=torch.long, device=device
            )
            alpha_input = torch.cat([start_tok, alpha_config[:, :-1]], dim=1)
            alpha_repr = self._run_alpha(alpha_input)

            # --- Sample beta channel ---
            self._clear_cache()
            beta_config = self._sample_channel(
                beta_config,
                n_electrons=self.n_beta,
                pos_embed=self.pos_embed_beta,
                blocks=self.beta_blocks,
                head=self.beta_head,
                ln=self.ln_beta,
                temperature=temperature,
                cross_kv=alpha_repr,
            )
        finally:
            self._disable_cache()

        return torch.cat([alpha_config, beta_config], dim=-1)

    def _sample_channel(
        self,
        config: torch.Tensor,
        n_electrons: int,
        pos_embed: torch.Tensor,
        blocks: nn.ModuleList,
        head: nn.Linear,
        ln: nn.LayerNorm,
        temperature: float,
        cross_kv: torch.Tensor | None,
    ) -> torch.Tensor:
        """Autoregressively sample one spin channel with particle conservation.

        Parameters
        ----------
        config : torch.Tensor
            Pre-allocated config tensor to fill, shape
            ``(n_samples, n_orbitals)``.
        n_electrons : int
            Exact number of electrons to place.
        pos_embed : torch.Tensor
            Positional embeddings for this channel.
        blocks : nn.ModuleList
            Transformer blocks for this channel.
        head : nn.Linear
            Output head producing logits.
        ln : nn.LayerNorm
            Layer norm before the output head.
        temperature : float
            Sampling temperature.
        cross_kv : torch.Tensor or None
            Cross-attention key/value from the other channel (alpha repr
            for beta sampling, ``None`` for alpha sampling).

        Returns
        -------
        torch.Tensor
            Filled configuration, shape ``(n_samples, n_orbitals)``.
        """
        n_samples = config.shape[0]
        n_orb = self.n_orbitals
        device = config.device

        electrons_placed = torch.zeros(n_samples, dtype=torch.long, device=device)
        start_tok = torch.full(
            (n_samples, 1), fill_value=2, dtype=torch.long, device=device
        )

        for pos in range(n_orb):
            # Current token: start token for pos=0, else previous occupation
            if pos == 0:
                current_tok = start_tok
            else:
                current_tok = config[:, pos - 1 : pos]  # (n_samples, 1)

            # Embed and add positional encoding
            h = self.token_embed(current_tok) + pos_embed[:, pos : pos + 1, :]

            # Run through transformer blocks
            for block in blocks:
                h = block(h, cross_kv=cross_kv)

            # Get logit for this position
            logit = head(ln(h)).squeeze(-1).squeeze(-1)  # (n_samples,)

            # Apply temperature
            if temperature != 1.0:
                logit = logit / temperature

            # Compute occupation probability
            prob_occupied = torch.sigmoid(logit)

            # Enforce particle conservation constraints
            remaining_positions = n_orb - pos
            electrons_needed = n_electrons - electrons_placed

            # Must occupy: not enough remaining positions for remaining electrons
            must_occupy = electrons_needed >= remaining_positions
            # Cannot occupy: already placed all electrons
            cannot_occupy = electrons_needed <= 0

            # Clamp probabilities
            prob_occupied = torch.where(
                must_occupy,
                torch.ones_like(prob_occupied),
                prob_occupied,
            )
            prob_occupied = torch.where(
                cannot_occupy,
                torch.zeros_like(prob_occupied),
                prob_occupied,
            )

            # Sample
            occupation = torch.bernoulli(prob_occupied).long()
            config[:, pos] = occupation
            electrons_placed = electrons_placed + occupation

        return config

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass --- delegates to :meth:`log_prob`.

        Parameters
        ----------
        alpha : torch.Tensor
            Alpha occupations, shape ``(batch, n_orbitals)``.
        beta : torch.Tensor
            Beta occupations, shape ``(batch, n_orbitals)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities, shape ``(batch,)``.
        """
        return self.log_prob(alpha, beta)
