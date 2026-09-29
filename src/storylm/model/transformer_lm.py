"""Configurable decoder-only Transformer language model."""

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from storylm.model.embedding import Embedding
from storylm.model.identity import Identity
from storylm.model.linear import Linear
from storylm.model.rmsnorm import RMSNorm
from storylm.model.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """Configurable decoder-only Transformer language model.

    Maps token IDs to embeddings, processes them through a stack of
    Transformer blocks, optionally applies final RMS normalization, and
    projects the hidden representations to vocabulary logits.

    Supports architecture ablations for RMSNorm placement, rotary positional
    embeddings, and the feed-forward network type.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        use_rmsnorm: bool = True,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        """Initialize the Transformer language model.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            context_length: Maximum sequence length.
            d_model: Dimensionality of hidden representations.
            num_layers: Number of Transformer blocks.
            num_heads: Number of attention heads.
            d_ff: Hidden dimensionality of the feed-forward network.
            theta: Base value for rotary positional embeddings.
            device: Device on which parameters are stored.
            dtype: Data type used for model parameters.
            use_rmsnorm: Whether to use RMS normalization.
            norm_style: Normalization placement, either "pre" or "post".
            use_rope: Whether to use rotary positional embeddings.
            ffn_type: Feed-forward architecture, either "swiglu" or "silu".
        """
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.context_length = context_length
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                    use_rmsnorm=use_rmsnorm,
                    norm_style=norm_style,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )

        if use_rmsnorm and norm_style == "pre":
            self.final_norm = RMSNorm(
                d_model,
                device=device,
                dtype=dtype,
            )
        else:
            self.final_norm = Identity()

        self.output_projection = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "... sequence_length vocab_size"]:
        """Compute next-token logits for an input token sequence.

        Args:
             in_indices: Token IDs for the input sequence.

        Returns:
            Unnormalized logits over the vocabulary at each sequence position.
        """
        x = self.token_embeddings(in_indices)

        for transformer_block in self.layers:
            x = transformer_block(x)

        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits

    def load_transformer_lm_weights(
        self,
        weights: dict[str, Tensor],
    ) -> None:
        """Load reference Transformer LM weights into the model."""
        with torch.no_grad():
            self.token_embeddings.weight.copy_(weights["token_embeddings.weight"])
            self.final_norm.weight.copy_(weights["ln_final.weight"])
            self.output_projection.weight.copy_(weights["lm_head.weight"])

            for i, transformer_block in enumerate(self.layers):
                transformer_block.attention.q_proj.weight.copy_(weights[f"layers.{i}.attn.q_proj.weight"])
                transformer_block.attention.k_proj.weight.copy_(weights[f"layers.{i}.attn.k_proj.weight"])
                transformer_block.attention.v_proj.weight.copy_(weights[f"layers.{i}.attn.v_proj.weight"])
                transformer_block.attention.output_proj.weight.copy_(weights[f"layers.{i}.attn.output_proj.weight"])

                transformer_block.norm1.weight.copy_(weights[f"layers.{i}.ln1.weight"])
                transformer_block.norm2.weight.copy_(weights[f"layers.{i}.ln2.weight"])

                transformer_block.feed_forward_nn.w1.weight.copy_(weights[f"layers.{i}.ffn.w1.weight"])
                transformer_block.feed_forward_nn.w2.weight.copy_(weights[f"layers.{i}.ffn.w2.weight"])
                transformer_block.feed_forward_nn.w3.weight.copy_(weights[f"layers.{i}.ffn.w3.weight"])
