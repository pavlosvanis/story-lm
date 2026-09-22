import random

import numpy.typing as npt
import torch

from torch import Tensor
from jaxtyping import Int


def load_data(dataset: npt.NDArray, batch_size: int, context_length: int,  device: torch.device | str | None = None) -> tuple[
    Int[Tensor, "batch_size context_length"],
    Int[Tensor, "batch_size context_length"],
]:
    inputs = torch.empty(batch_size, context_length, device=device, dtype=torch.long)
    targets = torch.empty(batch_size, context_length, device=device, dtype=torch.long)

    for b in range(batch_size):
        seq_start = random.randint(0, len(dataset) - context_length - 1)
        inputs[b] = torch.from_numpy(dataset[seq_start: seq_start + context_length]).to(device, dtype=torch.long)
        targets[b] = torch.from_numpy(dataset[seq_start + 1: seq_start + context_length + 1]).to(device, dtype=torch.long)

    return inputs, targets



