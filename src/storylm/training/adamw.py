"""AdamW optimization with decoupled weight decay."""

import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """Implement the AdamW optimization algorithm.

    AdamW maintains first- and second-moment estimates of each parameter's
    gradients and applies decoupled weight decay separately from the
    moment-based parameter update.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """Initialize the AdamW optimizer.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            betas: Exponential decay rates for the first- and second-moment
                estimates.
            eps: Small constant for numerical stability.
            weight_decay: Coefficient for decoupled weight decay.

        Raises:
            ValueError: If any optimizer hyperparameter is outside its valid
                range.
        """
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

    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        """Perform a single AdamW optimization step.

        Updates each parameter with a gradient using bias-corrected first- and
        second-moment estimates and decoupled weight decay.

        Args:
            closure: Optional callable that reevaluates the model and returns
                the loss.

        Returns:
            The loss returned by the closure, if a closure is provided;
            otherwise, None.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                # Parameters without gradients have no optimizer update.
                if p.grad is None:
                    continue

                # Moment estimates and step counts are maintained per parameter.
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                t = state["t"]

                m = state["m"]

                v = state["v"]

                grad = p.grad.data

                # Fold both bias-correction factors into the learning rate.
                adjusted_lr = lr * (math.sqrt(1 - (beta_2**t)) / (1 - (beta_1**t)))

                # Apply weight decay separately from the adaptive gradient update.
                p.data -= lr * weight_decay * p.data

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad**2)

                p.data -= adjusted_lr * (m / (torch.sqrt(v) + eps))

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
