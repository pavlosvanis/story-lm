import torch
from torch import nn

from torch import Tensor
from jaxtyping import Float

from student.rmsnorm import RMSNorm
from student.rope import RotaryPositionalEmbedding
from student.multi_head_self_attention import CausalMultiheadSelfAttention
from student.positionwise_feedforward import SwiGLU, SiLUFFN


class TransformerBlock(nn.Module):
    """Configurable Transformer block for architecture ablations.

    Applies causal multi-head self-attention and a position-wise feed-forward
    network with residual connections. Supports pre-norm and post-norm
    architectures, optional RMSNorm and RoPE, and either SwiGLU or SiLU
    feed-forward networks.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None, use_rmsnorm: bool = True, norm_style: str = "pre", use_rope: bool = True, ffn_type: str = "swiglu"):
        """Initialize a configurable Transformer block.

        Args:
            d_model: Dimensionality of the block input and output representations.
            num_heads: Number of attention heads.
            d_ff: Hidden dimensionality of the feed-forward network.
            max_seq_len: Maximum sequence length supported by RoPE.
            theta: Base value used for rotary positional embeddings.
            device: Device on which module parameters are stored.
            dtype: Data type used for module parameters.
            use_rmsnorm: Whether to use RMS normalization.
            norm_style: Normalization placement, either "pre" or "post".
            use_rope: Whether to use rotary positional embeddings.
            ffn_type: Feed-forward architecture, either "swiglu" or "silu".

        Raises:
            ValueError: If `norm_style` or `ffn_type` is unsupported.
        """

        super().__init__()

        if norm_style not in {"pre", "post"}:
            raise ValueError(
                "norm_style must be either 'pre' or 'post'"
            )

        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError(
                "ffn_type must be either 'swiglu' or 'silu'"
            )

        self.use_rmsnorm = use_rmsnorm
        self.norm_style = norm_style

        if use_rmsnorm:
            self.norm1 = RMSNorm(
                d_model,
                device=device,
                dtype=dtype,
            )
            self.norm2 = RMSNorm(
                d_model,
                device=device,
                dtype=dtype,
            )
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        rope = None

        if use_rope:
            rope = RotaryPositionalEmbedding(
                theta,
                d_model // num_heads,
                max_seq_len,
                device=device,
            )

        self.attention = CausalMultiheadSelfAttention(
            d_model,
            num_heads,
            rope,
            device=device,
            dtype=dtype,
        )

        if ffn_type == "swiglu":
            self.feed_forward_nn = SwiGLU(
                d_model,
                d_ff,
                device=device,
                dtype=dtype,
            )
        else:
            self.feed_forward_nn = SiLUFFN(
                d_model,
                d_ff,
                device=device,
                dtype=dtype,
            )


    def forward(
            self,
            x: Float[Tensor, "... sequence_length d_model"],
    ) -> Float[Tensor, "... sequence_length d_model"]:
        """Apply the Transformer block to a sequence of hidden representations.

        Args:
            x: Input hidden representations.

        Returns:
            Hidden representations with the same shape as the input.
        """

        if not self.use_rmsnorm:
            y = x + self.attention(x)
            return y + self.feed_forward_nn(y)

        if self.norm_style == "pre":
            y = x + self.attention(self.norm1(x))
            return y + self.feed_forward_nn(self.norm2(y))

        y = self.norm1(x + self.attention(x))

        return self.norm2(y + self.feed_forward_nn(y))