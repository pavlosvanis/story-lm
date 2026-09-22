import torch
from torch import nn

from torch import Tensor
from jaxtyping import Float

from student.rmsnorm import RMSNorm
from student.rope import RotaryPositionalEmbedding
from student.multi_head_self_attention import CausalMultiheadSelfAttention
from student.positionwise_feedforward import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with causal self-attention and SwiGLU.

    Applies RMS normalization before both the attention and feed-forward
    sublayers, with a residual connection around each sublayer.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)

        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
        self.attention = CausalMultiheadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.feed_forward_nn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

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

        y = x + self.attention(self.norm1(x))

        return y + self.feed_forward_nn(self.norm2(y))