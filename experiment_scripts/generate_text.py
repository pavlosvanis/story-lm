"""Generate a story continuation using the exported TinyStories model."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from storylm.inference.decoding import decode
from storylm.model.transformer_lm import TransformerLM
from storylm.tokenization.tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "final_model"

WEIGHTS_PATH = MODEL_DIR / "weights.pt"
CONFIG_PATH = MODEL_DIR / "config.json"

VOCAB_PATH = ARTIFACTS_DIR / "tinystories_vocab.pkl"
MERGES_PATH = ARTIFACTS_DIR / "tinystories_merges.pkl"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "generation"

SPECIAL_TOKENS = ["<|endoftext|>"]


def get_device() -> str:
    """Select an available device for text generation."""
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def load_model(device: str) -> tuple[TransformerLM, dict]:
    """Load the exported configuration and model weights on the requested device."""
    with CONFIG_PATH.open() as f:
        config = json.load(f)

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        theta=config["theta"],
        use_rmsnorm=config["use_rmsnorm"],
        norm_style=config["norm_style"],
        use_rope=config["use_rope"],
        ffn_type=config["ffn_type"],
        device=device,
        dtype=torch.float32,
    )

    weights = torch.load(
        WEIGHTS_PATH,
        map_location=device,
    )

    model.load_state_dict(weights)
    model.eval()

    return model, config


def load_tokenizer() -> Tokenizer:
    """Load the tokenizer vocabulary and merges used by the exported model."""
    return Tokenizer.from_files(
        str(VOCAB_PATH),
        str(MERGES_PATH),
        SPECIAL_TOKENS,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line generation settings."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading model from: {WEIGHTS_PATH}")

    tokenizer = load_tokenizer()
    model, model_config = load_model(device)

    completion = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    generated_text = args.prompt + completion

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "generated_text.txt"
    generation_config_path = OUTPUT_DIR / "generation_config.json"

    output_path.write_text(generated_text)

    generation_config = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "device": device,
        "model": model_config,
    }

    with generation_config_path.open("w") as f:
        json.dump(generation_config, f, indent=4)

    print()
    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(args.prompt)

    print()
    print("=" * 80)
    print("GENERATED TEXT")
    print("=" * 80)
    print(generated_text)

    print()
    print(f"Saved generated text to: {output_path}")
    print(f"Saved generation settings to: {generation_config_path}")


if __name__ == "__main__":
    main()
