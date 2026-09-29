"""Global L2 gradient clipping across model parameters."""

from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip gradients using their global L2 norm.

    Computes the L2 norm across all parameter gradients and, if it exceeds
    `max_l2_norm`, scales every gradient by the same factor so that the
    resulting global norm is bounded by the threshold.

    Args:
        parameters: Model parameters whose gradients should be clipped.
        max_l2_norm: Maximum allowed global gradient norm.
        eps: Small constant used for numerical stability.
    """
    parameters = list(parameters)

    global_l2_norm = 0
    for p in parameters:
        if p.grad is None:
            continue

        global_l2_norm += torch.linalg.vector_norm(p.grad) ** 2

    global_l2_norm = torch.sqrt(global_l2_norm)

    if global_l2_norm <= max_l2_norm:
        return

    scale_factor = max_l2_norm / (global_l2_norm + eps)

    for p in parameters:
        if p.grad is None:
            continue

        p.grad *= scale_factor
