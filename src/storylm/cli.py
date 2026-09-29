"""Command-line interface for StoryLM."""

import argparse
from pathlib import Path

from storylm.inference.generator import StoryGenerator


def build_parser() -> argparse.ArgumentParser:
    """Define the StoryLM commands and generation options."""
    parser = argparse.ArgumentParser(
        prog="storylm",
        description="Generate story continuations with a Transformer built from scratch.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Continue a story prompt.")

    generate.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Text to continue (default: Once upon a time).",
    )
    generate.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing model and tokenizer artifacts (default: artifacts).",
    )
    generate.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, such as cpu, mps, or cuda; auto selects an available device.",
    )
    generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens (default: 256).",
    )
    generate.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Positive sampling temperature (default: 0.8).",
    )
    generate.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold in (0, 1] (default: 0.9).",
    )
    generate.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42).",
    )
    return parser


def main() -> None:
    """Load the selected artifacts and generate a story from CLI arguments."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        generator = StoryGenerator.from_artifacts(
            args.artifacts_dir,
            device=args.device,
        )
        result = generator.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print("PROMPT")
    print(result.prompt)
    print()
    print("GENERATED TEXT")
    print(result.full_text)


if __name__ == "__main__":
    main()
