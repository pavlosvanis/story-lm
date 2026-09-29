"""Autoregressive text generation with temperature and nucleus sampling."""

import torch

from storylm.model.softmax import softmax
from storylm.model.transformer_lm import TransformerLM
from storylm.tokenization.tokenizer import Tokenizer


def decode(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: torch.device | str | None = None,
) -> str:
    """Generate a text completion from a Transformer language model.

    Encode the prompt into token IDs and repeatedly sample a next token from
    the model's predicted distribution. Generation stops when the
    <|endoftext|> token is produced or when `max_new_tokens` tokens have been
    generated. Temperature scaling and top-p sampling are applied before each
    token is sampled.

    Args:
        model: Trained Transformer language model used for generation.
        tokenizer: Tokenizer used to encode the prompt and decode generated
            token IDs.
        prompt: Text prefix from which generation begins.
        max_new_tokens: Maximum number of tokens to generate after the prompt.
        temperature: Temperature used to scale next-token logits before
            applying softmax.
        top_p: Cumulative probability threshold used for nucleus sampling.
        device: Device on which generation is performed.

    Returns:
        The generated text completion, excluding the original prompt.

    Raises:
        ValueError: If `max_new_tokens` is negative, `temperature` is not
            positive, `top_p` is outside the interval (0, 1], or the prompt
            encodes to no tokens.
    """
    _validate_inputs(max_new_tokens, temperature, top_p)

    prompt_token_ids = tokenizer.encode(prompt)
    prompt_length = len(prompt_token_ids)

    if prompt_length == 0:
        raise ValueError("prompt must encode to at least one token")

    generated_sequence = torch.tensor(prompt_token_ids, device=device, dtype=torch.long).unsqueeze(
        0
    )  # (prompt_length,) -> (1, prompt_length)
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Restrict each forward pass to the model's context window.
            model_input = generated_sequence[:, -model.context_length :]
            logits = model(model_input).squeeze(0)  # (sequence_length, vocab_size)
            next_token_logits = logits[-1]

            # Scale logits before converting them to probabilities.
            next_token_probs = softmax(next_token_logits / temperature, dim=-1)

            next_token_id = _sample_top_p(next_token_probs, top_p)

            if tokenizer.decode([next_token_id]) == "<|endoftext|>":
                break

            next_token = generated_sequence.new_tensor([[next_token_id]])

            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

        all_token_ids = generated_sequence.squeeze(0).cpu().tolist()
        completion_tokens = all_token_ids[prompt_length:]
        text_completion = tokenizer.decode(completion_tokens)

    return text_completion


def _sample_top_p(
    probabilities: torch.Tensor,
    top_p: float,
) -> int:
    """Sample a token ID using nucleus sampling.

    Args:
        probabilities: Probability distribution over vocabulary tokens.
        top_p: Cumulative probability threshold for nucleus sampling.

    Returns:
        Sampled vocabulary token ID.
    """
    # Keep vocabulary token IDs aligned with sorted probabilities.
    sorted_probs, sorted_token_ids = torch.sort(probabilities, descending=True)
    cumulative_prob = 0
    cutoff_idx = len(sorted_probs) - 1

    # Retain the smallest prefix whose cumulative probability reaches top_p.
    for i in range(len(sorted_probs)):
        cumulative_prob += sorted_probs[i]
        if cumulative_prob >= top_p:
            cutoff_idx = i
            break

    nucleus_probs = sorted_probs[: cutoff_idx + 1]

    # Renormalize the retained probabilities before sampling.
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    sampled_position = torch.multinomial(nucleus_probs, num_samples=1).item()
    next_token_id = sorted_token_ids[sampled_position].item()

    return next_token_id


def _validate_inputs(max_new_tokens: int, temperature: float, top_p: float) -> None:
    """Validate decoding hyperparameters.

    Raises:
        ValueError: If any decoding hyperparameter is outside its valid range.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
