
import torch
from typing import Iterable


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float, eps: float = 1e-6
) -> None:
    parameters = list(parameters)

    global_l2_norm = 0
    for p in parameters:
        if p.grad is None:
            continue

        global_l2_norm += torch.linalg.vector_norm(p.grad) ** 2

    global_l2_norm = torch.sqrt(global_l2_norm)

    if global_l2_norm <= max_l2_norm:
        return

    scale_factor = (max_l2_norm / (global_l2_norm + eps))

    for p in parameters:
        if p.grad is None:
            continue

        p.grad *= scale_factor





