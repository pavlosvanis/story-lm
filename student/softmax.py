import math

import torch
from jaxtyping import Float
from torch import Tensor


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
