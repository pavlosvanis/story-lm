"""Reproduce Transformer architecture ablations and stability probes."""

import argparse
import random

import numpy as np
import torch

from experiment_scripts.experiment_config import (
    BASE_MAX_LEARNING_RATE,
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
    EXPERIMENTS_DIR,
    MAX_L2_NORM,
    MIN_LR_RATIO,
    NUM_EVAL_BATCHES,
    NUM_HEADS,
    NUM_ITERATIONS,
    NUM_LAYERS,
    PROJECT_ROOT,
    SEED,
    THETA,
    TRAINING_TOKENS_PATH,
    VALIDATION_TOKENS_PATH,
    VOCAB_SIZE,
    WARMUP_ITERS,
    WEIGHT_DECAY,
)
from storylm.training import training_loop
from storylm.training.experiment_utils import save_experiment_results

ARCHITECTURES = {
    "post_norm": {
        "use_rmsnorm": True,
        "norm_style": "post",
        "use_rope": True,
        "ffn_type": "swiglu",
        "d_ff": D_FF,
    },
    "nope": {
        "use_rmsnorm": True,
        "norm_style": "pre",
        "use_rope": False,
        "ffn_type": "swiglu",
        "d_ff": D_FF,
    },
    "silu": {
        "use_rmsnorm": True,
        "norm_style": "pre",
        "use_rope": True,
        "ffn_type": "silu",
        "d_ff": 4 * D_MODEL,
    },
    "no_rmsnorm": {
        "use_rmsnorm": False,
        "norm_style": "pre",
        "use_rope": True,
        "ffn_type": "swiglu",
        "d_ff": D_FF,
    },
}


def format_learning_rate(learning_rate: float) -> str:
    """Format a learning rate for an experiment directory name."""
    return f"{learning_rate:.0e}"


def get_experiment_dir(
    experiment_name: str,
    num_iterations: int,
    max_learning_rate: float,
):
    """Return the output directory for an architecture experiment."""
    if experiment_name == "no_rmsnorm":
        lr_name = format_learning_rate(max_learning_rate)

        if num_iterations != NUM_ITERATIONS:
            return EXPERIMENTS_DIR / "architecture" / "probes" / (f"no_rmsnorm_lr_{lr_name}_steps_{num_iterations}")

        return EXPERIMENTS_DIR / "architecture" / "no_rmsnorm" / f"lr_{lr_name}"

    if num_iterations != NUM_ITERATIONS:
        return EXPERIMENTS_DIR / "architecture" / "probes" / f"{experiment_name}_steps_{num_iterations}"

    return EXPERIMENTS_DIR / "architecture" / experiment_name


def run_architecture_experiment(
    experiment_name: str,
    num_iterations: int,
    max_learning_rate: float,
) -> None:
    """Run one Transformer architecture ablation."""
    if experiment_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture experiment: {experiment_name}")

    if not TRAINING_TOKENS_PATH.exists():
        raise FileNotFoundError(f"Training tokens not found: {TRAINING_TOKENS_PATH}")

    if not VALIDATION_TOKENS_PATH.exists():
        raise FileNotFoundError(f"Validation tokens not found: {VALIDATION_TOKENS_PATH}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    architecture = ARCHITECTURES[experiment_name]

    experiment_dir = get_experiment_dir(experiment_name, num_iterations, max_learning_rate)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results_path = experiment_dir / "results.json"

    is_full_run = num_iterations == NUM_ITERATIONS

    if is_full_run:
        checkpoint_path = experiment_dir / "checkpoint.pt"
    else:
        checkpoint_path = None

    if results_path.exists():
        raise FileExistsError(f"Experiment results already exist: {results_path}")

    if checkpoint_path is not None and checkpoint_path.exists():
        raise FileExistsError(f"Experiment checkpoint already exists: {checkpoint_path}")

    min_learning_rate = MIN_LR_RATIO * max_learning_rate

    # Architecture experiments use the same schedule as the baseline.
    # A short probe therefore represents the first N steps of the
    # corresponding full 5000-step experiment.
    warmup_iters = WARMUP_ITERS
    cosine_cycle_iters = COSINE_CYCLE_ITERS

    if is_full_run:
        eval_interval = EVAL_INTERVAL
        checkpoint_interval = CHECKPOINT_INTERVAL
    else:
        # About ten validation measurements for a diagnostic probe.
        eval_interval = max(
            1,
            num_iterations // 10,
        )
        checkpoint_interval = num_iterations

    total_tokens = BATCH_SIZE * num_iterations * CONTEXT_LENGTH

    print(f"\n{'=' * 60}")
    print(f"Architecture experiment: {experiment_name}")
    print(f"{'=' * 60}")
    print(f"Training data:     {TRAINING_TOKENS_PATH}")
    print(f"Validation data:   {VALIDATION_TOKENS_PATH}")
    print(f"Results dir:       {experiment_dir}")
    print(f"RMSNorm:           {architecture['use_rmsnorm']}")
    print(f"Norm style:        {architecture['norm_style']}")
    print(f"RoPE:              {architecture['use_rope']}")
    print(f"FFN:               {architecture['ffn_type']}")
    print(f"d_ff:              {architecture['d_ff']}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Training steps:    {num_iterations:,}")
    print(f"Training tokens:   {total_tokens:,}")
    print(f"Eval batch size:   {BATCH_SIZE}")
    print(f"Warmup steps:      {warmup_iters}")
    print(f"Cosine end:        {cosine_cycle_iters}")
    print(f"Eval interval:     {eval_interval}")
    print(f"Max LR:            {max_learning_rate}")

    model, train_losses, validation_losses, eval_steps, eval_times = training_loop.train_model(
        vocab_size=VOCAB_SIZE,
        training_tokens_path=str(TRAINING_TOKENS_PATH),
        validation_tokens_path=str(VALIDATION_TOKENS_PATH),
        num_iterations=num_iterations,
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=architecture["d_ff"],
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
        eval_batch_size=BATCH_SIZE,
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=(str(checkpoint_path) if checkpoint_path is not None else None),
        use_rmsnorm=architecture["use_rmsnorm"],
        norm_style=architecture["norm_style"],
        use_rope=architecture["use_rope"],
        ffn_type=architecture["ffn_type"],
    )

    config = {
        "experiment_name": experiment_name,
        "experiment_type": "architecture_ablation",
        "is_full_run": is_full_run,
        "training_tokens_path": str(TRAINING_TOKENS_PATH),
        "validation_tokens_path": str(VALIDATION_TOKENS_PATH),
        "vocab_size": VOCAB_SIZE,
        "num_iterations": num_iterations,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": BATCH_SIZE,
        "total_training_tokens": total_tokens,
        "context_length": CONTEXT_LENGTH,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "d_ff": architecture["d_ff"],
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
        "use_rmsnorm": architecture["use_rmsnorm"],
        "norm_style": architecture["norm_style"],
        "use_rope": architecture["use_rope"],
        "ffn_type": architecture["ffn_type"],
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


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=ARCHITECTURES.keys(),
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=NUM_ITERATIONS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=BASE_MAX_LEARNING_RATE,
    )

    args = parser.parse_args()

    run_architecture_experiment(experiment_name=args.experiment, num_iterations=args.steps, max_learning_rate=args.lr)


if __name__ == "__main__":
    main()
