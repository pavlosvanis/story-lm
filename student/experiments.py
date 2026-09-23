from pathlib import Path

import torch

from student import training_loop
from experiment_utils import save_experiment_results

experiment_name = "lr_1e-3"

experiment_dir = Path("experiments") / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)

training_tokens_path = "..."
validation_tokens_path = "..."

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
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.0

# Gradient clipping
max_l2_norm = 1.0

# Learning-rate schedule
max_learning_rate = 1e-3

# We will choose/tune these before the real experiment.
min_learning_rate = ...
warmup_iters = ...
cosine_cycle_iters = num_iterations

# Validation
eval_interval = ...
num_eval_batches = ...

# Checkpointing
checkpoint_interval = 1000
checkpoint_path = experiment_dir / "checkpoint.pt"

# Device
device = "mps"
dtype = torch.float32

model, train_losses, validation_losses, eval_steps, eval_times = (
    training_loop.train_model(
        vocab_size=vocab_size,
        training_tokens_path=training_tokens_path,
        validation_tokens_path=validation_tokens_path,
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

    "training_tokens_path": training_tokens_path,
    "validation_tokens_path": validation_tokens_path,

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
}

save_experiment_results(
    str(experiment_dir / "results.json"),
    config,
    train_losses,
    validation_losses,
    eval_steps,
    eval_times,
)
