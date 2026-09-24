import random

import numpy as np
import torch

from experiment_scripts.experiment_config import (
    EXPERIMENTS_DIR,
    TRAINING_TOKENS_PATH,
    VALIDATION_TOKENS_PATH,
    SEED,
    BATCH_SIZE,
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    D_MODEL,
    NUM_LAYERS,
    NUM_HEADS,
    D_FF,
    THETA,
    BETAS,
    EPS,
    WEIGHT_DECAY,
    MAX_L2_NORM,
    WARMUP_ITERS,
    MIN_LR_RATIO,
    COSINE_CYCLE_ITERS,
    EVAL_INTERVAL,
    NUM_EVAL_BATCHES,
    CHECKPOINT_INTERVAL,
    DEVICE,
    DTYPE,
)

from student import training_loop
from student.experiment_utils import save_experiment_results


PROBE_NUM_ITERATIONS = 500
PROBE_MAX_LEARNING_RATE = 1e-1


def run_divergence_experiment(max_learning_rate: float) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    experiment_name = f"lr_{max_learning_rate:.0e}"

    experiment_dir = (
        EXPERIMENTS_DIR
        / "learning_rate"
        / "divergence"
        / experiment_name
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if not TRAINING_TOKENS_PATH.exists():
        raise FileNotFoundError(
            f"Training tokens not found: {TRAINING_TOKENS_PATH}"
        )

    if not VALIDATION_TOKENS_PATH.exists():
        raise FileNotFoundError(
            f"Validation tokens not found: {VALIDATION_TOKENS_PATH}"
        )

    results_path = experiment_dir / "results.json"
    checkpoint_path = experiment_dir / "checkpoint.pt"

    if results_path.exists() or checkpoint_path.exists():
        raise FileExistsError(
            f"Experiment output already exists: {experiment_dir}"
        )

    min_learning_rate = MIN_LR_RATIO * max_learning_rate

    print(f"Training data:   {TRAINING_TOKENS_PATH}")
    print(f"Validation data: {VALIDATION_TOKENS_PATH}")
    print(f"Results dir:     {experiment_dir}")
    print(f"Probe steps:     {PROBE_NUM_ITERATIONS}")
    print(f"Max LR:          {max_learning_rate}")
    print(f"Min LR:          {min_learning_rate}")
    print(f"Warmup steps:    {WARMUP_ITERS}")
    print(f"Cosine end:      {COSINE_CYCLE_ITERS}")

    model, train_losses, validation_losses, eval_steps, eval_times = (
        training_loop.train_model(
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
    )

    if validation_losses:
        print(
            f"\nFinal validation loss at step {eval_steps[-1]}: "
            f"{validation_losses[-1]:.4f}"
        )

    del model

    if DEVICE == "mps":
        torch.mps.empty_cache()


def run_divergence_probe() -> None:
    print(
        f"\nStarting divergence probe with "
        f"max LR = {PROBE_MAX_LEARNING_RATE}\n"
    )

    run_divergence_experiment(PROBE_MAX_LEARNING_RATE)


if __name__ == "__main__":
    run_divergence_probe()