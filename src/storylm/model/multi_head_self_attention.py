"""Causal multi-head attention with optional rotary positional embeddings."""

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn

from storylm.model.linear import Linear
from storylm.model.rope import RotaryPositionalEmbedding
from storylm.model.scaled_dot_product_attention import scaled_dot_product_attention


class CausalMultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional rotary embeddings.

    Projects the input into query, key, and value representations, splits
    them across attention heads, optionally applies rotary positional
    embeddings to queries and keys, performs causal scaled dot-product
    attention, and projects the concatenated head outputs back to the model
    dimension.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize attention projections and optional positional rotations."""
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        X: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        """Apply causal multi-head self-attention.

        Args:
            X: Input hidden representations.
            token_positions: Optional token positions used by rotary positional
                embeddings. If omitted, positions start from zero.

        Returns:
            Attention output with the same shape as the input.
        """
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        Q_h = rearrange(
            Q,
            "... queries (num_heads d_head) -> ... num_heads queries d_head",
            num_heads=self.num_heads,
        )

        K_h = rearrange(
            K,
            "... keys (num_heads d_head) -> ... num_heads keys d_head",
            num_heads=self.num_heads,
        )

        V_h = rearrange(
            V,
            "... values (num_heads d_head) -> ... num_heads values d_head",
            num_heads=self.num_heads,
        )

        seq_len = X.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=X.device))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=X.device)

            rope_positions = token_positions.unsqueeze(-2)

            Q_h = self.rope(Q_h, rope_positions)
            K_h = self.rope(K_h, rope_positions)

        Z_h = scaled_dot_product_attention(Q_h, K_h, V_h, mask)

        Z = rearrange(Z_h, "... num_heads queries d_head  -> ... queries (num_heads d_head)")

        return self.output_proj(Z)
