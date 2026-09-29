"""Generate a story continuation using the exported TinyStories model."""

import argparse
import json
from pathlib import Path

from storylm.inference.generator import StoryGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "final_model"

WEIGHTS_PATH = MODEL_DIR / "weights.pt"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "generation"


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

    print(f"Loading model from: {WEIGHTS_PATH}")

    generator = StoryGenerator.from_artifacts(ARTIFACTS_DIR)
    print(f"Using device: {generator.device}")

    result = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    generated_text = result.full_text

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
        "device": generator.device,
        "model": generator.model_config,
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
