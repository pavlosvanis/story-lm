import torch
from torch import nn

from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        """Learned lookup table mapping token IDs to embedding vectors."""

        super().__init__()

        self.num_embeddings = num_embeddings  # Vocabulary size
        self.embedding_dim = embedding_dim  # Size of each token embedding

        # One learned embedding vector per vocabulary token
        embedding_matrix = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = nn.Parameter(embedding_matrix)  # self.weight[token_id] = token id's embedding vector

        # Initialize embeddings from N(0, 1), truncated to [-3, 3].
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        """Look up embedding vectors for the provided token IDs.

        Args:
            token_ids: Integer token IDs.

        Returns:
            Learned embedding vectors with an additional embedding dimension.
        """

        return self.weight[token_ids]
