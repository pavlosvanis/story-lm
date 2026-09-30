"""Prepare, train, and export the selected TinyStories model."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from storylm.data.preparation import prepare_tinystories
from storylm.inference.generator import StoryGenerator

from .config import MODEL_CONFIG, SEED, TOTAL_TOKEN_BUDGET, TRAINING_CONFIG
from .training_loop import train_model


@dataclass(frozen=True)
class TrainingResult:
    """Location of a completed model and whether it was reused."""

    model_dir: Path
    reused: bool


def train_tinystories(
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    *,
    device: str = "auto",
) -> TrainingResult:
    """Reuse a completed model or train and export the selected configuration.

    Existing generation artifacts are loaded to validate them before reuse.
    Reuse does not require raw text or token arrays. For a new model, prepare
    missing data and save checkpoints, history, weights, and configuration
    under final_model. Interrupted runs leave their outputs for recovery.

    Args:
        data_dir: Directory for downloaded TinyStories text.
        artifacts_dir: Directory containing tokenizer and model artifacts.
            Choose a separate directory to train another model.
        device: PyTorch device name, or "auto" to select MPS, CUDA, or CPU.

    Returns:
        The model directory and whether an existing model was reused.

    Raises:
        ValueError: If existing artifacts are incomplete or invalid.
        OSError: If preparing data or saving the model fails.
    """
    artifacts_dir = Path(artifacts_dir).expanduser().resolve()
    model_dir = artifacts_dir / "final_model"
    weights_path = model_dir / "weights.pt"
    config_path = model_dir / "config.json"

    if weights_path.is_file() and config_path.is_file():
        StoryGenerator.from_artifacts(artifacts_dir, device=device)
        print(f"Reusing existing model: {model_dir}")
        return TrainingResult(model_dir=model_dir, reused=True)

    if model_dir.exists():
        raise ValueError(
            f"Model output already exists but is incomplete: {model_dir}. "
            "Recover the existing run or choose a separate artifacts directory."
        )

    prepared = prepare_tinystories(data_dir=data_dir, artifacts_dir=artifacts_dir)

    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Reserve the output directory without replacing any existing run.
    model_dir.mkdir()
    checkpoint_path = model_dir / "checkpoint.pt"
    history_path = model_dir / "history.json"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Training device: {device}")
    print(f"Training steps: {TRAINING_CONFIG['num_iterations']:,}")
    print(f"Training token budget: {TOTAL_TOKEN_BUDGET:,}")
    print(f"Model output: {model_dir}")

    model, training_losses, validation_losses, eval_steps, eval_times = train_model(
        **MODEL_CONFIG,
        **TRAINING_CONFIG,
        training_tokens_path=str(prepared.training_tokens_path),
        validation_tokens_path=str(prepared.validation_tokens_path),
        device=device,
        dtype=torch.float32,
        checkpoint_path=str(checkpoint_path),
    )

    history = [
        {
            "step": step,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "elapsed_seconds": elapsed,
        }
        for step, training_loss, validation_loss, elapsed in zip(
            eval_steps, training_losses, validation_losses, eval_times, strict=True
        )
    ]
    record = {
        "config": {
            **MODEL_CONFIG,
            **TRAINING_CONFIG,
            "training_tokens_path": prepared.training_tokens_path.name,
            "validation_tokens_path": prepared.validation_tokens_path.name,
            "total_token_budget": TOTAL_TOKEN_BUDGET,
            "seed": SEED,
            "device": device,
            "dtype": "torch.float32",
        },
        "history": history,
    }
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write("\n")

    exported_config = {
        **MODEL_CONFIG,
        "source": "storylm train",
        "final_validation_loss": validation_losses[-1],
    }
    torch.save(model.state_dict(), weights_path)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(exported_config, file, indent=2, allow_nan=False)
        file.write("\n")

    print(f"Saved model weights: {weights_path}")
    print(f"Saved model configuration: {config_path}")
    print(f"Saved training history: {history_path}")
    return TrainingResult(model_dir=model_dir, reused=False)
