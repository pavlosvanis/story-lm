import math
from typing import Optional, Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta_1: {betas[0]}")

        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta_2: {betas[1]}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # go through every trainable parameter in this group
            for p in group["params"]:

                # gradient gets populated by loss.backward()
                if p.grad is None:
                    continue

                # self.state[p] is a dictionary containing persistent optimizer
                # information associated specifically with parameter p.
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                # iteration t
                t = state["t"]

                # first moment m
                m = state["m"]

                # second moment v
                v = state["v"]

                grad = p.grad.data  # gradient of the loss wrt to this parameter p

                adjusted_lr = lr * (math.sqrt(1 - (beta_2 ** t)) / (
                        1 - (beta_1 ** t)))  # compute adjusted learning rate for iteration t

                p.data -= lr * weight_decay * p.data  # apply decoupled weight decay for regularization, shrinking parameter towards zero

                m = beta_1 * m + (1 - beta_1) * grad  # update first moment estimate
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)  # update second moment estimate

                p.data -= adjusted_lr * (m / (torch.sqrt(v) + eps))  # apply moment-adjusted weight updates

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
