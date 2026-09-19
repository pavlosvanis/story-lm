import math

import torch
from torch import nn
from jaxtyping import Float
from torch import Tensor
from einops import einsum


class Linear(nn.Module):
    """Linear transformation y = Wx without a bias term."""
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Allocate the weight matrix with shape (out_features, in_features).
        # torch.empty does not initialize the tensor values.
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)

        # Register the weight matrix as a learnable model parameter.
        self.weight = nn.Parameter(weight)

        std = math.sqrt(2.0 / (in_features + out_features))

        # Initialize the existing weight parameter in place from a normal distribution
        # centered at 0, truncated so all values lie within three standard deviations.
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")




