"""Command-line interface for StoryLM."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Define the StoryLM commands and their options."""
    parser = argparse.ArgumentParser(
        prog="storylm",
        description="Prepare data, train a Transformer from scratch, and generate stories.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare or reuse TinyStories data and tokenizer artifacts.")
    prepare.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for downloaded raw text files (default: data).",
    )
    prepare.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for tokenizer files and token arrays (default: artifacts).",
    )

    train = commands.add_parser("train", help="Train the selected TinyStories model or reuse existing artifacts.")
    train.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for downloaded raw text files (default: data).",
    )
    train.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for data and model artifacts; use a separate directory for another training run.",
    )
    train.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, such as cpu, mps, or cuda; auto selects an available device.",
    )

    gui = commands.add_parser("gui", help="Open the desktop story writer.")
    gui.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing model and tokenizer artifacts (default: artifacts).",
    )
    gui.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, such as cpu, mps, or cuda; auto selects an available device.",
    )

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
    """Run the selected StoryLM command."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            _prepare(args)
        elif args.command == "train":
            _train(args)
        elif args.command == "gui":
            _gui(args)
        elif args.command == "generate":
            _generate(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


def _prepare(args: argparse.Namespace) -> None:
    """Prepare data and report the artifact directory."""
    from storylm.data.preparation import prepare_tinystories

    prepared = prepare_tinystories(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
    )
    print(f"Prepared artifacts: {prepared.vocab_path.parent}")


def _train(args: argparse.Namespace) -> None:
    """Train or reuse the selected model and report its directory."""
    from storylm.training.workflow import train_tinystories

    trained = train_tinystories(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        device=args.device,
    )
    print(f"Model ready: {trained.model_dir}")


def _gui(args: argparse.Namespace) -> None:
    """Open the desktop writer with the selected generation artifacts."""
    from storylm.gui import launch_gui

    launch_gui(args.artifacts_dir, device=args.device)


def _generate(args: argparse.Namespace) -> None:
    """Load the selected artifacts and print a story continuation."""
    from storylm.inference.generator import StoryGenerator

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

    print("PROMPT")
    print(result.prompt)
    print()
    print("GENERATED TEXT")
    print(result.full_text)


if __name__ == "__main__":
    main()
