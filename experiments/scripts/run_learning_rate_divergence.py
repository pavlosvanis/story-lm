"""Probe training stability at large learning rates."""

import random

import numpy as np
import torch

from experiments.config import (
    BATCH_SIZE,
    BETAS,
    CHECKPOINT_INTERVAL,
    CONTEXT_LENGTH,
    COSINE_CYCLE_ITERS,
    D_FF,
    D_MODEL,
    DEVICE,
    DTYPE,
    EPS,
    EVAL_INTERVAL,
    MAX_L2_NORM,
    MIN_LR_RATIO,
    NUM_EVAL_BATCHES,
    NUM_HEADS,
    NUM_LAYERS,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    THETA,
    TRAINING_TOKENS_PATH,
    VALIDATION_TOKENS_PATH,
    VOCAB_SIZE,
    WARMUP_ITERS,
    WEIGHT_DECAY,
)
from experiments.save_results import save_experiment_results
from storylm.training import training_loop

PROBE_NUM_ITERATIONS = 500
PROBE_MAX_LEARNING_RATE = 1e-1


def run_divergence_experiment(max_learning_rate: float) -> None:
    """Run one learning-rate stability probe."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    experiment_name = f"lr_{max_learning_rate:.0e}"

    experiment_dir = RESULTS_DIR / "learning_rate" / "divergence" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if not TRAINING_TOKENS_PATH.exists():
        raise FileNotFoundError(f"Training tokens not found: {TRAINING_TOKENS_PATH}")

    if not VALIDATION_TOKENS_PATH.exists():
        raise FileNotFoundError(f"Validation tokens not found: {VALIDATION_TOKENS_PATH}")

    results_path = experiment_dir / "results.json"
    checkpoint_path = experiment_dir / "checkpoint.pt"

    if results_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"Experiment output already exists: {experiment_dir}")

    min_learning_rate = MIN_LR_RATIO * max_learning_rate

    print(f"Training data:   {TRAINING_TOKENS_PATH}")
    print(f"Validation data: {VALIDATION_TOKENS_PATH}")
    print(f"Results dir:     {experiment_dir}")
    print(f"Probe steps:     {PROBE_NUM_ITERATIONS}")
    print(f"Max LR:          {max_learning_rate}")
    print(f"Min LR:          {min_learning_rate}")
    print(f"Warmup steps:    {WARMUP_ITERS}")
    print(f"Cosine end:      {COSINE_CYCLE_ITERS}")

    model, train_losses, validation_losses, eval_steps, eval_times = training_loop.train_model(
        vocab_size=VOCAB_SIZE,
        training_tokens_path=str(TRAINING_TOKENS_PATH),
        validation_tokens_path=str(VALIDATION_TOKENS_PATH),
        num_iterations=PROBE_NUM_ITERATIONS,
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        theta=THETA,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        max_l2_norm=MAX_L2_NORM,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=WARMUP_ITERS,
        cosine_cycle_iters=COSINE_CYCLE_ITERS,
        device=DEVICE,
        dtype=DTYPE,
        eval_interval=EVAL_INTERVAL,
        num_eval_batches=NUM_EVAL_BATCHES,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        checkpoint_path=str(checkpoint_path),
    )

    config = {
        "experiment_name": experiment_name,
        "experiment_type": "learning_rate_divergence_probe",
        "training_tokens_path": str(TRAINING_TOKENS_PATH),
        "validation_tokens_path": str(VALIDATION_TOKENS_PATH),
        "vocab_size": VOCAB_SIZE,
        "num_iterations": PROBE_NUM_ITERATIONS,
        "batch_size": BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "d_ff": D_FF,
        "theta": THETA,
        "betas": BETAS,
        "eps": EPS,
        "weight_decay": WEIGHT_DECAY,
        "max_l2_norm": MAX_L2_NORM,
        "max_learning_rate": max_learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": WARMUP_ITERS,
        "cosine_cycle_iters": COSINE_CYCLE_ITERS,
        "eval_interval": EVAL_INTERVAL,
        "num_eval_batches": NUM_EVAL_BATCHES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "seed": SEED,
    }

    save_experiment_results(
        str(results_path),
        config,
        train_losses,
        validation_losses,
        eval_steps,
        eval_times,
        project_root=PROJECT_ROOT,
    )

    if validation_losses:
        print(f"\nFinal validation loss at step {eval_steps[-1]}: {validation_losses[-1]:.4f}")

    del model

    if DEVICE == "mps":
        torch.mps.empty_cache()


def run_divergence_probe() -> None:
    """Compare large learning rates using the reference model configuration."""
    print(f"\nStarting divergence probe with max LR = {PROBE_MAX_LEARNING_RATE}\n")

    run_divergence_experiment(PROBE_MAX_LEARNING_RATE)


if __name__ == "__main__":
    run_divergence_probe()
