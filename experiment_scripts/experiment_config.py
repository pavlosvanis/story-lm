from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

TRAINING_TOKENS_PATH = ARTIFACTS_DIR / "tinystories_train_tokens.npy"
VALIDATION_TOKENS_PATH = ARTIFACTS_DIR / "tinystories_valid_tokens.npy"

SEED = 42

# Training
NUM_ITERATIONS = 5000
BATCH_SIZE = 32

# Model
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
THETA = 10_000

# Low-resource training budget
TOTAL_TOKEN_BUDGET = BATCH_SIZE * NUM_ITERATIONS * CONTEXT_LENGTH

# AdamW
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1

# Gradient clipping
MAX_L2_NORM = 1.0

# Learning-rate schedule
WARMUP_ITERS = int(0.02 * NUM_ITERATIONS)
MIN_LR_RATIO = 0.1  # scale minimum LR relative to maximum LR
COSINE_CYCLE_ITERS = NUM_ITERATIONS - 1

BASE_MAX_LEARNING_RATE = 3e-3 # found from lr_sweep experiment

# Validation
EVAL_INTERVAL = 100
NUM_EVAL_BATCHES = 10

# Checkpointing
CHECKPOINT_INTERVAL = 1000

# Device
DEVICE = "mps"
DTYPE = torch.float32
