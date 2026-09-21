import torch
from torch import nn
from student import linear

from torch import Tensor
from jaxtyping import Float


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)  # weight: (d_ff, d_model)
        self.w2 = linear.Linear(d_ff, d_model, device=device, dtype=dtype)  # weight: (d_model, d_ff)
        self.w3 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)  # weight: (d_ff, d_model)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        w1_x = self.w1(x)
        w3_x = self.w3(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        # Gate the SiLU-activated w1 branch elementwise with the w3 branch
        swiglu = self.w2(silu_w1_x * w3_x)

        return swiglu
