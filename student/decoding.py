import torch


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

    Encodes the prompt into token IDs and repeatedly samples a next token from
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
        The prompt followed by the generated text completion.

    Raises:
        ValueError: If `max_new_tokens` is negative, `temperature` is not
            positive, or `top_p` is outside the interval (0, 1].
    """