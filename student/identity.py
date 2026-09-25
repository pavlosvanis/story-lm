# student/identity.py

from torch import nn
from torch import Tensor


class Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x