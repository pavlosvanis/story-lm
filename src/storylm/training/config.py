"""Selected TinyStories model and training settings for normal training."""

SEED = 42
TOTAL_TOKEN_BUDGET = 40_960_000
BATCH_SIZE = 64
CONTEXT_LENGTH = 256

NUM_ITERATIONS = TOTAL_TOKEN_BUDGET // (BATCH_SIZE * CONTEXT_LENGTH)
MAX_LEARNING_RATE = 3e-3

MODEL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": CONTEXT_LENGTH,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1344,
    "theta": 10_000,
    "use_rmsnorm": True,
    "norm_style": "pre",
    "use_rope": True,
    "ffn_type": "swiglu",
}

TRAINING_CONFIG = {
    "num_iterations": NUM_ITERATIONS,
    "batch_size": BATCH_SIZE,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.1,
    "max_l2_norm": 1.0,
    "max_learning_rate": MAX_LEARNING_RATE,
    "min_learning_rate": 0.1 * MAX_LEARNING_RATE,
    "warmup_iters": max(1, int(0.02 * NUM_ITERATIONS)),
    "cosine_cycle_iters": NUM_ITERATIONS - 1,
    "eval_interval": max(1, NUM_ITERATIONS // 50),
    "num_eval_batches": 10,
    "eval_batch_size": 32,
    "checkpoint_interval": max(1, NUM_ITERATIONS // 5),
}
