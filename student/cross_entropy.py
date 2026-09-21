import torch

from torch import Tensor
from jaxtyping import Float
from jaxtyping import Int


def cross_entropy(
        logits: Float[Tensor, "... vocab_size"],  # (batch, seq_len, vocab_size) -> one logit per vocabulary item
        targets: Int[Tensor, "..."],  # correct next-token ID for each example/position (batch, seq_len)
) -> Float[Tensor, ""]:
    # Shift each logit vector by its maximum value for numerical stability.
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values

    # Select the logit assigned to the correct target at each position.
    target_logits = torch.gather(
        logits,
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(
        -1)  # For each (i, j), picks logits[i, j, targets[i, j]] -> (batch, seq_len, 1) -> then squeeze into (batch, seq_len)

    # compute the negative log-likelihood for each target
    loss_per_token = -target_logits + torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))  # (batch,seq_len)

    return loss_per_token.mean()  # average across all examples and positions
