import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float
from jaxtyping import Int

from student.embedding import Embedding
from student.transformer_block import TransformerBlock
from student.rmsnorm import RMSNorm
from student.linear import Linear
class TransformerLM(nn.Module):
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
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_projection = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
            self,
            in_indices: Int[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "... sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)

        for transformer_block in self.layers:
            x = transformer_block(x)

        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits





