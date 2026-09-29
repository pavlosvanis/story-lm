"""Position-wise SwiGLU and SiLU feed-forward networks."""

import torch
from jaxtyping import Float
from torch import Tensor, nn

from storylm.model import linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply the SiLU activation elementwise."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU position-wise feed-forward network.

    Applies a SiLU-activated projection gated elementwise by a second
    projection, followed by a projection back to the model dimension.
    """

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """Initialize the gated input projections and output projection."""
        super().__init__()
        self.w1 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)  # weight: (d_ff, d_model)
        self.w2 = linear.Linear(d_ff, d_model, device=device, dtype=dtype)  # weight: (d_model, d_ff)
        self.w3 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)  # weight: (d_ff, d_model)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """Apply the SwiGLU feed-forward transformation.

        Args:
            x: Input hidden representations.

        Returns:
            Transformed representations with the same final dimension as the input.
        """
        w1_x = self.w1(x)
        w3_x = self.w3(x)
        silu_w1_x = silu(w1_x)
        # Gate the SiLU-activated w1 branch elementwise with the w3 branch.
        swiglu = self.w2(silu_w1_x * w3_x)

        return swiglu


class SiLUFFN(nn.Module):
    """Position-wise feed-forward network using SiLU without gating."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the input and output projections."""
        super().__init__()

        self.w1 = linear.Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )
        self.w2 = linear.Linear(
            d_ff,
            d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        """Apply the SiLU feed-forward transformation."""
        return self.w2(silu(self.w1(x)))
