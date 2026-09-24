import time
import numpy as np
import torch
from .data_loading import load_data
from .transformer_lm import TransformerLM
from .cross_entropy import cross_entropy
from .adamw import AdamW
from .checkpointing import save_checkpoint
from .gradient_clipping import gradient_clipping
from .learning_rate_schedule import learning_rate_schedule


def train_model(
        # data
        vocab_size: int,
        training_tokens_path: str,
        validation_tokens_path: str,

        # training
        num_iterations: int,
        batch_size: int,

        # model
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,

        # AdamW
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,

        # gradient clipping
        max_l2_norm: float = 1.0,

        # learning-rate schedule
        max_learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-4,
        warmup_iters: int = 100,
        cosine_cycle_iters: int = 1000,

        # device
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,

        # validation
        eval_interval: int = 100,
        num_eval_batches: int = 10,
        eval_batch_size: int | None = None,

        # checkpoints
        checkpoint_interval: int = 1000,
        checkpoint_path: str | None = None,
) -> tuple[TransformerLM, list[float], list[float], list[int], list[float]]:
    """Train a Transformer language model and periodically evaluate it.

     Loads tokenized training and validation datasets using memory mapping,
     trains the model with AdamW, applies gradient clipping and a learning-rate
     schedule, periodically evaluates validation loss, and optionally saves
     checkpoints.

     Args:
         vocab_size: Number of tokens in the vocabulary.
         training_tokens_path: Path to the tokenized training dataset.
         validation_tokens_path: Path to the tokenized validation dataset.
         num_iterations: Number of optimization steps to perform.
         batch_size: Number of token sequences per training batch.
         context_length: Number of tokens in each input sequence.
         d_model: Dimensionality of the model representations.
         num_layers: Number of Transformer blocks.
         num_heads: Number of attention heads.
         d_ff: Hidden dimensionality of the feed-forward network.
         theta: Base value used for rotary positional embeddings.
         betas: Exponential decay rates for AdamW moment estimates.
         eps: Numerical stability constant used by AdamW.
         weight_decay: AdamW decoupled weight-decay coefficient.
         max_l2_norm: Maximum allowed global gradient norm.
         max_learning_rate: Maximum learning rate used by the schedule.
         min_learning_rate: Minimum learning rate used by the schedule.
         warmup_iters: Number of learning-rate warmup steps.
         cosine_cycle_iters: Number of steps in the cosine schedule.
         device: Device on which training is performed.
         dtype: Floating-point dtype used for model parameters.
         eval_interval: Number of training steps between evaluations.
         num_eval_batches: Number of validation batches averaged per evaluation.
         eval_batch_size: Number of sequences per validation batch.
         checkpoint_interval: Number of training steps between checkpoints.
         checkpoint_path: Path at which checkpoints are saved, if provided.

     Returns:
         A tuple containing the trained model, average training losses,
         average validation losses, the training steps at which evaluation
         was performed, and the corresponding elapsed wall-clock times.

     Raises:
         ValueError: If the training configuration or dataset shape is invalid.
     """

    _validate_training_config(vocab_size, num_iterations, batch_size, context_length, d_model, num_layers, num_heads,
                              d_ff, theta, max_l2_norm, max_learning_rate, min_learning_rate, warmup_iters,
                              cosine_cycle_iters, eval_interval, num_eval_batches, checkpoint_interval)

    training_tokens = np.load(training_tokens_path, mmap_mode="r")
    validation_tokens = np.load(validation_tokens_path, mmap_mode="r")

    _validate_dataset(
        training_tokens,
        context_length,
        "training"
    )

    _validate_dataset(
        validation_tokens,
        context_length,
        "validation"
    )

    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta, device, dtype)
    optimizer = AdamW(model.parameters(), max_learning_rate, betas, eps, weight_decay)

    avg_training_losses = []
    avg_validation_losses = []
    eval_steps = []
    eval_times = []

    running_train_loss = 0.0

    if eval_batch_size is None:
        eval_batch_size = batch_size

    start_time = time.perf_counter()
    for i in range(num_iterations):
        # inputs and targets each have shape (batch_size, context_length)
        inputs, targets = load_data(training_tokens, batch_size, context_length, device)

        optimizer.zero_grad()  # clear gradients from the previous optimization step
        logits = model(inputs)  # forward pass: (batch_size, context_length) -> (batch_size, context_length, vocab_size)

        loss = cross_entropy(logits, targets)  # compute next-token prediction loss

        # accumulate training loss over the current evaluation interval
        running_train_loss += loss.item()

        loss.backward()  # backpropagation, compute gradients for all trainable parameters
        gradient_clipping(model.parameters(), max_l2_norm)  # clip global gradient norm
        lr = learning_rate_schedule(i, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()  # update weights

        # Validation
        if (i + 1) % eval_interval == 0:  # every eval_interval training steps
            avg_validation_loss = _evaluate(model, validation_tokens, eval_batch_size, context_length, device,
                                            num_eval_batches)

            avg_validation_losses.append(avg_validation_loss)

            # average training loss over the training steps since the previous evaluation
            avg_training_loss = running_train_loss / eval_interval
            avg_training_losses.append(avg_training_loss)

            eval_steps.append(i + 1)
            elapsed_time = time.perf_counter() - start_time
            eval_times.append(elapsed_time)
            running_train_loss = 0.0  # reset training-loss accumulator for the next evaluation interval

            print(
                f"step {i + 1}: "
                f"avg train loss over last {eval_interval} steps={avg_training_loss:.4f}, "
                f"avg validation loss over {num_eval_batches} batches={avg_validation_loss:.4f}, "
                f"lr={lr:.6g}, "
                f"time={elapsed_time:.1f}s"
            )

        # Checkpointing
        if (i + 1) % checkpoint_interval == 0:  # every checkpoint_interval training steps
            if checkpoint_path:
                save_checkpoint(model, optimizer, i + 1, checkpoint_path)

    # Save final checkpoint of model
    if checkpoint_path:
        save_checkpoint(model, optimizer, num_iterations, checkpoint_path)

    return model, avg_training_losses, avg_validation_losses, eval_steps, eval_times


def _validate_training_config(
        vocab_size: int,
        num_iterations: int,
        batch_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_l2_norm: float,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
        eval_interval: int,
        num_eval_batches: int,
        checkpoint_interval: int
) -> None:
    """Validate model and training hyperparameters.

    Raises:
        ValueError: If any configuration value is invalid.
    """

    # training arguments
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # model arguments
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    if d_model <= 0:
        raise ValueError("d_model must be positive")

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    if num_heads <= 0:
        raise ValueError("num_heads must be positive")

    if d_ff <= 0:
        raise ValueError("d_ff must be positive")

    if theta <= 0:
        raise ValueError("theta must be positive")

    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    d_head = d_model // num_heads
    if d_head % 2 != 0:
        raise ValueError("d_model / num_heads must be even for RoPE")

    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    if max_learning_rate <= 0:
        raise ValueError("max_learning_rate must be positive")

    if min_learning_rate < 0:
        raise ValueError("min_learning_rate cannot be negative")

    if min_learning_rate > max_learning_rate:
        raise ValueError("min_learning_rate cannot exceed max_learning_rate")

    if warmup_iters < 0:
        raise ValueError("warmup_iters cannot be negative")

    if cosine_cycle_iters <= warmup_iters:
        raise ValueError("cosine_cycle_iters must be greater than warmup_iters")

    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")

    if num_eval_batches <= 0:
        raise ValueError("num_eval_batches must be positive")

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")


def _validate_dataset(
        tokens: np.ndarray,
        context_length: int,
        dataset_name: str) -> None:
    """Validate the structure and length of a tokenized dataset.

    Args:
        tokens: One-dimensional sequence of token IDs.
        context_length: Number of tokens required for each model input.
        dataset_name: Name used to identify the dataset in error messages.

    Raises:
        ValueError: If the dataset is not one-dimensional or is too short.
    """

    if tokens.ndim != 1:
        raise ValueError(f"{dataset_name} dataset must be a 1D sequence of token IDs")

    if len(tokens) <= context_length:
        raise ValueError(f"{dataset_name} dataset must contain more than context_length tokens")


def _evaluate(model: TransformerLM, validation_tokens: np.ndarray, batch_size: int, context_length: int,
              device: torch.device | str | None, num_eval_batches: int) -> float:
    """Estimate validation loss for the current model.

    The loss is averaged across multiple randomly sampled validation batches to
    reduce noise from any single batch.

    Args:
        model: Model to evaluate.
        validation_tokens: Tokenized validation dataset.
        batch_size: Number of sequences per validation batch.
        context_length: Number of tokens in each sequence.
        device: Device on which validation is performed.
        num_eval_batches: Number of validation batches to average.

    Returns:
        Average validation cross-entropy loss.
    """

    model.eval()
    validation_loss_sum = 0.0

    # disable gradient tracking during validation since no parameter updates are performed
    with torch.no_grad():
        # average over multiple randomly sampled validation batches to reduce
        # noise from evaluating on a single batch
        for _ in range(num_eval_batches):
            validation_inputs, validation_targets = load_data(validation_tokens, batch_size, context_length,
                                                              device)

            validation_logits = model(validation_inputs)
            validation_loss_sum += cross_entropy(validation_logits, validation_targets).item()

    model.train()

    # average validation loss across several randomly sampled validation batches
    # to obtain a less noisy estimate of the current model's validation performance
    avg_validation_loss = validation_loss_sum / num_eval_batches

    return avg_validation_loss
