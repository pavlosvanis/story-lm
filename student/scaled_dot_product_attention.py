import math

from einops import einsum
from torch import Tensor

from jaxtyping import Float
from jaxtyping import Bool

from student.softmax import softmax


def scaled_dot_product_attention(
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    attention_scores = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys")
    d_k = Q.shape[-1]
    pre_softmax_attention_weights = attention_scores / math.sqrt(d_k)

    if mask is not None:
        # we mask where entry is False, so we need to negate mask
        pre_softmax_attention_weights = pre_softmax_attention_weights.masked_fill(~mask, float('-inf'))

    attention_weights = softmax(pre_softmax_attention_weights, dim=-1)

    return attention_weights @ V
