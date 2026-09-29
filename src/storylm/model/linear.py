"""Bias-free linear projections implemented with tensor contractions."""

import math

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn


class Linear(nn.Module):
    """Linear transformation y = Wx without a bias term."""

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """Initialize a bias-free linear projection."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight layout matches PyTorch: (out_features, in_features).
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)

        self.weight = nn.Parameter(weight)

        std = math.sqrt(2.0 / (in_features + out_features))

        # Use a zero-mean normal, truncated at three standard deviations.
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        """Apply the projection across the final input dimension."""
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
