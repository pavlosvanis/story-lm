import random

import numpy.typing as npt
import torch

from torch import Tensor
from jaxtyping import Int


def load_data(dataset: npt.NDArray, batch_size: int, context_length: int,  device: torch.device | str | None = None) -> tuple[
    Int[Tensor, "batch_size context_length"],
    Int[Tensor, "batch_size context_length"],
]:
    """Sample a batch of next-token prediction sequences.

    Each input sequence is a contiguous window from the tokenized dataset.
    The corresponding target sequence is the same window shifted forward by
    one token.

    Args:
        dataset: One-dimensional sequence of token IDs.
        batch_size: Number of sequences to sample.
        context_length: Number of input tokens in each sequence.
        device: Device on which the returned tensors are placed.

    Returns:
        A pair containing input and target token tensors, each with shape
        (batch_size, context_length).
    """

    inputs = torch.empty(batch_size, context_length, device=device, dtype=torch.long)
    targets = torch.empty(batch_size, context_length, device=device, dtype=torch.long)

    for b in range(batch_size):
        seq_start = random.randint(0, len(dataset) - context_length - 1)
        input_slice = dataset[seq_start: seq_start + context_length].copy()
        target_slice = dataset[seq_start + 1: seq_start + context_length + 1].copy()

        inputs[b] = torch.from_numpy(input_slice).to(device, dtype=torch.long)
        targets[b] = torch.from_numpy(target_slice).to(device, dtype=torch.long)

    return inputs, targets



