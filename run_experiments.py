from pathlib import Path
import random
import numpy as np
import torch

from student import training_loop
from student.experiment_utils import save_experiment_results

PROJECT_ROOT = Path(__file__).resolve().parent

def run_experiment(max_learning_rate: float) -> None:
    seed = 42  # same initialization and batch sequence across experiments

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    experiment_name = f"lr_{max_learning_rate:.0e}"

    experiment_dir = PROJECT_ROOT / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    training_tokens_path = PROJECT_ROOT / "artifacts" / "tinystories_train_tokens.npy"
    validation_tokens_path = PROJECT_ROOT / "artifacts" / "tinystories_valid_tokens.npy"

    if not training_tokens_path.exists():
        raise FileNotFoundError(
            f"Training tokens not found: {training_tokens_path}"
        )

    if not validation_tokens_path.exists():
        raise FileNotFoundError(
            f"Validation tokens not found: {validation_tokens_path}"
        )

    print(f"Training data:   {training_tokens_path}")
    print(f"Validation data: {validation_tokens_path}")
    print(f"Results dir:     {experiment_dir}")

    # Training
    num_iterations = 5000
    batch_size = 32

    # Model
    vocab_size = 10_000
    context_length = 256
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    theta = 10_000

    # AdamW
    betas = (0.9, 0.95)
    eps = 1e-8
    weight_decay = 0.1

    # Gradient clipping
    max_l2_norm = 1.0

    # Learning-rate schedule -> VARY LR for experiment
    max_learning_rate = max_learning_rate

    # We will choose/tune these before the real experiment.
    min_learning_rate = 0.1 * max_learning_rate
    warmup_iters = int(0.02 * num_iterations)  # 100 steps
    cosine_cycle_iters = num_iterations - 1

    # Validation
    eval_interval = 100
    num_eval_batches = 10

    # Checkpointing
    checkpoint_interval = 1000
    checkpoint_path = experiment_dir / "checkpoint.pt"

    # Device
    device = "mps"
    dtype = torch.float32

    model, train_losses, validation_losses, eval_steps, eval_times = (
        training_loop.train_model(
            vocab_size=vocab_size,
            training_tokens_path=str(training_tokens_path),
            validation_tokens_path=str(validation_tokens_path),
            num_iterations=num_iterations,
            batch_size=batch_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            theta=theta,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_l2_norm=max_l2_norm,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
            device=device,
            dtype=dtype,
            eval_interval=eval_interval,
            num_eval_batches=num_eval_batches,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=str(checkpoint_path),
        )
    )

    config = {
        "experiment_name": experiment_name,

        "training_tokens_path": str(training_tokens_path),
        "validation_tokens_path": str(validation_tokens_path),

        "vocab_size": vocab_size,
        "num_iterations": num_iterations,
        "batch_size": batch_size,

        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "theta": theta,

        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "max_l2_norm": max_l2_norm,

        "max_learning_rate": max_learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": cosine_cycle_iters,

        "eval_interval": eval_interval,
        "num_eval_batches": num_eval_batches,

        "checkpoint_interval": checkpoint_interval,

        "device": str(device),
        "dtype": str(dtype),

        "seed": seed
    }

    save_experiment_results(
        str(experiment_dir / "results.json"),
        config,
        train_losses,
        validation_losses,
        eval_steps,
        eval_times,
    )

    del model
    torch.mps.empty_cache()


if __name__ == "__main__":
    # LR sweep: keep architecture, data, seed, optimizer settings, and schedule shape fixed across runs.
    # Use beta2=0.95 and a 2% warmup as fixed baseline settings because of training run being short (5000 steps)
    # Test approximately log-spaced learning rates, with 3e-3 as an aggressive value to probe instability/divergence.
    learning_rates = [1e-4, 3e-4, 6e-4, 1e-3, 3e-3]

    for learning_rate in learning_rates:
        print(f"\nStarting LR experiment: {learning_rate}\n")

        try:
            run_experiment(learning_rate)
        except Exception as e:
            print(f"Experiment with LR {learning_rate} failed: {e}")
            print("Moving to the next learning rate...")
        finally:
            torch.mps.empty_cache()
