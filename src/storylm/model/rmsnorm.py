"""Root mean square normalization of hidden representations."""

import torch
from jaxtyping import Float
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root mean square normalization over the model dimension.

    Normalizes each hidden vector by its root mean square and applies a
    learned elementwise gain.
    """

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """Initialize the learned gain and normalization epsilon."""
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        gain = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(gain)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """Normalize input vectors across their final dimension.

        Args:
            x: Input hidden representations.

        Returns:
            RMS-normalized representations in the input dtype.
        """
        in_dtype = x.dtype

        # Accumulate squared activations in float32 to reduce overflow risk.
        x = x.to(torch.float32)

        # Normalize each token independently while preserving batch dimensions.
        mean_square = torch.mean(x**2, dim=-1, keepdim=True)

        rms = torch.sqrt(mean_square + self.eps)
        rms_norm = (x / rms) * self.weight

        return rms_norm.to(in_dtype)
