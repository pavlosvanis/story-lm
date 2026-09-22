import math

import torch
from jaxtyping import Float
from torch import Tensor


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """Compute a numerically stable softmax along a specified dimension.

    Args:
        x: Input tensor.
        dim: Dimension over which probabilities are normalized.

    Returns:
        Tensor of normalized probabilities with the same shape as `x`.
    """

    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
