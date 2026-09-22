import torch
from torch import nn
from einops import einsum
from einops import rearrange

from torch import Tensor
from jaxtyping import Int
from jaxtyping import Float


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary positional embeddings to query or key vectors.

    Precomputes sine and cosine values for all supported sequence positions
    and rotates adjacent pairs of hidden dimensions according to each token's
    position.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # Create all possible token positions and RoPE dimension-pair indices
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        k = torch.arange(d_k // 2, dtype=torch.float32, device=device)

        # Compute one rotation frequency for each pair of hidden dimensions.
        # k is zero-indexed, so this corresponds to the assignment's (2k - 2) term
        omega = theta ** (-(2 * k) / d_k)  # zero indexed

        # angles[i][k] gives us the rotation angle of the kth 2d vector in token vector at position i in the sequence
        angles = einsum(positions, omega, "token_position, pair -> token_position pair")

        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        # Store the precomputed values as buffers because they belong to the
        # module and should move with it across devices, but are not trainable.
        # persistent=False excludes them from the state_dict because they can
        # always be recomputed from theta, d_k, and max_seq_len.
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: Float[Tensor, " ... seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        """Apply rotary position-dependent rotations to the input vectors.

        Args:
            x: Query or key vectors whose final dimension is partitioned into
                adjacent two-dimensional pairs.
            token_positions: Sequence positions corresponding to the tokens in `x`.

        Returns:
            Positionally rotated vectors with the same shape as `x`.
        """

        # For each token in x, select the precomputed cosine and sine values
        # corresponding to its actual sequence position.
        # Shape: (..., seq_len, d_k // 2), one value per 2D pair.
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # Split each d_k-dimensional query/key vector into adjacent 2D pairs:
        # [x0, x1, x2, x3, ...] -> [[x0, x1], [x2, x3], ...]
        # Shape: (..., seq_len, d_k) -> (..., seq_len, d_k // 2, 2)
        x_pairs = rearrange(
            x,
            "... seq_len (pair two) -> ... seq_len pair two",
            two=2,
        )

        # Extract the first and second coordinate of every 2D pair.
        # Both have shape (..., seq_len, d_k // 2).
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]

        # Apply the 2D rotation to every token and every pair in parallel:
        # [a']   [ cos  -sin ] [a]
        # [b'] = [ sin   cos ] [b]
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos

        # Reconstruct each rotated 2D pair:
        # rotated_a = [a0', a1', ...]
        # rotated_b = [b0', b1', ...]
        # -> [[a0', b0'], [a1', b1'], ...]
        rotated_pairs = torch.stack((rotated_a, rotated_b), dim=-1)
        rotated_x = rearrange(rotated_pairs, "... seq_len pair two -> ... seq_len (pair two)")

        return rotated_x










