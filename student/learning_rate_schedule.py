import math


def learning_rate_schedule(
    t: int,
    max_lr: float,
    min_lr: float,
    t_warmup: int,
    t_cosine: int,
) -> float:

    if t < t_warmup:
        return (t / t_warmup) * max_lr
    elif t <= t_cosine:
        return min_lr + 0.5 * (1 + math.cos(math.pi * (t - t_warmup) / (t_cosine - t_warmup))) * (max_lr - min_lr)
    else:
        return min_lr

