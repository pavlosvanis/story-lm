"""Compare batch sizes under a fixed training-token budget."""

import random

import numpy as np
import torch

from experiments.config import (
    BASE_MAX_LEARNING_RATE,
    BETAS,
    CONTEXT_LENGTH,
    D_FF,
    D_MODEL,
    DEVICE,
    DTYPE,
    EPS,
    MAX_L2_NORM,
    MIN_LR_RATIO,
    NUM_EVAL_BATCHES,
    NUM_HEADS,
    NUM_LAYERS,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    THETA,
    TOTAL_TOKEN_BUDGET,
    TRAINING_TOKENS_PATH,
    VALIDATION_TOKENS_PATH,
    VOCAB_SIZE,
    WEIGHT_DECAY,
)
from experiments.config import (
    BATCH_SIZE as BASE_BATCH_SIZE,
)
from experiments.save_results import save_experiment_results
from storylm.training import training_loop

BATCH_SIZES = [1, 16, 64, 112]


def run_batch_size_experiment(batch_size: int) -> None:
    """Train one batch-size configuration under the fixed token budget."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    experiment_name = f"batch_{batch_size}"

    experiment_dir = RESULTS_DIR / "batch_size" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if not TRAINING_TOKENS_PATH.exists():
        raise FileNotFoundError(f"Training tokens not found: {TRAINING_TOKENS_PATH}")

    if not VALIDATION_TOKENS_PATH.exists():
        raise FileNotFoundError(f"Validation tokens not found: {VALIDATION_TOKENS_PATH}")

    results_path = experiment_dir / "results.json"
    checkpoint_path = experiment_dir / "checkpoint.pt"

    if results_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"Experiment output already exists: {experiment_dir}")

    max_learning_rate = BASE_MAX_LEARNING_RATE
    min_learning_rate = MIN_LR_RATIO * max_learning_rate

    num_iterations = TOTAL_TOKEN_BUDGET // (batch_size * CONTEXT_LENGTH)

    warmup_iters = max(1, int(0.02 * num_iterations))

    cosine_cycle_iters = num_iterations - 1

    # Aim for roughly 50 validation measurements per run.
    eval_interval = max(1, num_iterations // 50)

    # Aim for roughly 5 periodic checkpoints per run.
    checkpoint_interval = max(1, num_iterations // 5)

    total_tokens = batch_size * num_iterations * CONTEXT_LENGTH

    print(f"Training data:     {TRAINING_TOKENS_PATH}")
    print(f"Validation data:   {VALIDATION_TOKENS_PATH}")
    print(f"Results dir:       {experiment_dir}")

    print(f"Batch size:        {batch_size}")
    print(f"Training steps:    {num_iterations:,}")
    print(f"Training tokens:   {total_tokens:,}")
    print(f"Eval batch size:   {BASE_BATCH_SIZE}")
    print(f"Warmup steps:      {warmup_iters}")
    print(f"Cosine end:        {cosine_cycle_iters}")
    print(f"Eval interval:     {eval_interval}")
    print(f"Checkpoint every:  {checkpoint_interval}")
    print(f"Max LR:            {max_learning_rate}")

    model, train_losses, validation_losses, eval_steps, eval_times = training_loop.train_model(
        vocab_size=VOCAB_SIZE,
        training_tokens_path=str(TRAINING_TOKENS_PATH),
        validation_tokens_path=str(VALIDATION_TOKENS_PATH),
        num_iterations=num_iterations,
        batch_size=batch_size,
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
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
        device=DEVICE,
        dtype=DTYPE,
        eval_interval=eval_interval,
        num_eval_batches=NUM_EVAL_BATCHES,
        eval_batch_size=BASE_BATCH_SIZE,
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=str(checkpoint_path),
    )

    config = {
        "experiment_name": experiment_name,
        "experiment_type": "batch_size",
        "training_tokens_path": str(TRAINING_TOKENS_PATH),
        "validation_tokens_path": str(VALIDATION_TOKENS_PATH),
        "vocab_size": VOCAB_SIZE,
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "eval_batch_size": BASE_BATCH_SIZE,
        "total_token_budget": TOTAL_TOKEN_BUDGET,
        "total_training_tokens": total_tokens,
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
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": cosine_cycle_iters,
        "eval_interval": eval_interval,
        "num_eval_batches": NUM_EVAL_BATCHES,
        "checkpoint_interval": checkpoint_interval,
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
        print(f"\nFinal validation loss: {validation_losses[-1]:.4f}")

    del model

    if DEVICE == "mps":
        torch.mps.empty_cache()


def run_batch_size_sweep() -> None:
    """Run the configured batch-size comparisons."""
    for batch_size in BATCH_SIZES:
        print(f"\n{'=' * 60}\nStarting batch-size experiment: {batch_size}\n{'=' * 60}\n")

        try:
            run_batch_size_experiment(batch_size)

        except FileExistsError as error:
            print(f"Skipping existing experiment: {error}")

        except RuntimeError as error:
            error_message = str(error).lower()

            if "out of memory" in error_message or "mps backend out of memory" in error_message:
                print(f"Batch size {batch_size} exceeded device memory.")
                continue

            raise

        finally:
            if DEVICE == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    run_batch_size_sweep()
