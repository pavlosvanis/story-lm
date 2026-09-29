"""Serialization of model, optimizer, and training iteration state."""

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """Save model, optimizer, and training iteration state.

    Args:
        model: Model whose parameters should be saved.
        optimizer: Optimizer whose state should be saved.
        iteration: Training iteration associated with the checkpoint.
        out: File path or binary file-like object to which the checkpoint
            is written.
    """
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    """Restore model and optimizer state from a checkpoint.

    Args:
        src: File path or binary file-like object containing the checkpoint.
        model: Model into which the saved parameters are loaded.
        optimizer: Optimizer into which the saved state is loaded.

    Returns:
        The training iteration stored in the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
