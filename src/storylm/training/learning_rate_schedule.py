"""Linear learning-rate warmup followed by cosine decay."""

import math


def learning_rate_schedule(
    t: int,
    max_lr: float,
    min_lr: float,
    t_warmup: int,
    t_cosine: int,
) -> float:
    """Compute the learning rate for a training step.

    Uses linear warmup from zero to `max_lr`, followed by cosine decay from
    `max_lr` to `min_lr`, and then keeps the learning rate at `min_lr`.

    Args:
        t: Current training step.
        max_lr: Maximum learning rate reached after warmup.
        min_lr: Minimum learning rate after cosine decay.
        t_warmup: Number of warmup steps.
        t_cosine: Step at which cosine decay reaches `min_lr`.

    Returns:
        Learning rate for step `t`.
    """
    if t < t_warmup:
        return (t / t_warmup) * max_lr
    elif t <= t_cosine:
        return min_lr + 0.5 * (1 + math.cos(math.pi * (t - t_warmup) / (t_cosine - t_warmup))) * (max_lr - min_lr)
    else:
        return min_lr
