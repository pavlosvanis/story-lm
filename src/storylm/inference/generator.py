"""Reusable story generation with exported model and tokenizer artifacts."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch

from storylm.model.transformer_lm import TransformerLM
from storylm.tokenization.tokenizer import Tokenizer

from .decoding import decode


@dataclass(frozen=True)
class GenerationResult:
    """A prompt and its generated continuation."""

    prompt: str
    completion: str

    @property
    def full_text(self) -> str:
        """Return the prompt followed by its completion."""
        return self.prompt + self.completion


class StoryGenerator:
    """Reuse a loaded model and tokenizer across generation calls."""

    def __init__(
        self,
        model: TransformerLM,
        tokenizer: Tokenizer,
        model_config: dict,
        device: str,
    ) -> None:
        """Store the loaded model, tokenizer, configuration, and device."""
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str | Path = "artifacts",
        *,
        device: str = "auto",
    ) -> Self:
        """Load the exported model and its tokenizer once.

        Args:
            artifacts_dir: Directory containing the tokenizer files and
                the final_model subdirectory. Relative paths use the current
                working directory.
            device: PyTorch device name, or "auto" to select MPS, CUDA, or CPU
                in that order.

        Returns:
            A generator ready to produce story continuations.

        Raises:
            FileNotFoundError: If any required artifact is missing.
        """
        artifacts_dir = Path(artifacts_dir).expanduser().resolve()
        config_path = artifacts_dir / "final_model" / "config.json"
        weights_path = artifacts_dir / "final_model" / "weights.pt"
        vocab_path = artifacts_dir / "tinystories_vocab.pkl"
        merges_path = artifacts_dir / "tinystories_merges.pkl"

        missing = [str(path) for path in (config_path, weights_path, vocab_path, merges_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing generation artifacts:\n" + "\n".join(missing))

        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        with config_path.open(encoding="utf-8") as file:
            model_config = json.load(file)

        tokenizer = Tokenizer.from_files(
            str(vocab_path),
            str(merges_path),
            ["<|endoftext|>"],
        )
        model = TransformerLM(
            vocab_size=model_config["vocab_size"],
            context_length=model_config["context_length"],
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            theta=model_config["theta"],
            use_rmsnorm=model_config["use_rmsnorm"],
            norm_style=model_config["norm_style"],
            use_rope=model_config["use_rope"],
            ffn_type=model_config["ffn_type"],
            device=device,
            dtype=torch.float32,
        )
        weights = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(weights)
        model.eval()

        return cls(model, tokenizer, model_config, device)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        seed: int = 42,
    ) -> GenerationResult:
        """Generate a continuation using the already loaded model.

        Args:
            prompt: Text to continue.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Positive scaling factor for next-token logits.
            top_p: Cumulative probability threshold for nucleus sampling.
            seed: Seed applied to the global Python, NumPy, and PyTorch random
                generators before decoding each request.

        Returns:
            The original prompt and generated completion.

        Raises:
            ValueError: If the prompt or decoding settings are invalid.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        completion = decode(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=self.device,
        )
        return GenerationResult(prompt=prompt, completion=completion)
