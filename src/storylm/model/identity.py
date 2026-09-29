"""Identity module used by architecture ablations."""

from torch import Tensor, nn


class Identity(nn.Module):
    """Pass inputs through without changing them."""

    def forward(self, x: Tensor) -> Tensor:
        """Return the input tensor."""
        return x
