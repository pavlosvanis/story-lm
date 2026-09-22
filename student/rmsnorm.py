import torch
from torch import nn

from jaxtyping import Float
from torch import Tensor


class RMSNorm(nn.Module):
    """Root mean square normalization over the model dimension.

       Normalizes each hidden vector by its root mean square and applies a
       learned elementwise gain.
       """

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
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

        # Upcast input to prevent overflow when squaring
        x = x.to(torch.float32)

        # Compute mean_square for hidden dimension (last dimension) for each token while keeping the dimension
        # Normalize each token's hidden vector independently across d_model.
        # Leading dimensions such as batch and sequence are processed in parallel.
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)

        rms = torch.sqrt(mean_square + self.eps)
        rms_norm = (x / rms) * self.weight

        return rms_norm.to(in_dtype)  # return result in the original input dtype
