"""Portable serialization of experiment configurations and loss histories."""

import json
from pathlib import Path


def save_experiment_results(
    out_path: str,
    config: dict,
    training_losses: list[float],
    validation_losses: list[float],
    eval_steps: list[int],
    eval_times: list[float],
    *,
    project_root: str | Path,
) -> None:
    """Save evaluation history with a portable experiment configuration.

    Paths within the project are recorded relative to ``project_root``.
    Absolute paths outside the project are recorded by filename only so
    exported results do not contain local account or directory names.
    The caller's configuration is not modified.

    Args:
        out_path: Destination for the JSON experiment record.
        config: Experiment settings, including any fields ending in ``_path``.
        training_losses: Mean training loss at each evaluation.
        validation_losses: Mean validation loss at each evaluation.
        eval_steps: Training steps corresponding to the recorded losses.
        eval_times: Elapsed seconds corresponding to the recorded losses.
        project_root: Directory against which project paths are made relative.
    """
    portable_config = dict(config)
    root = Path(project_root).resolve()
    for key, value in config.items():
        if key.endswith("_path") and isinstance(value, (str, Path)):
            config_path = Path(value)
            if config_path.is_absolute():
                try:
                    config_path = config_path.relative_to(root)
                except ValueError:
                    config_path = Path(config_path.name)
            portable_config[key] = config_path.as_posix()

    history = [
        {
            "step": step,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "elapsed_seconds": elapsed_time,
        }
        for step, training_loss, validation_loss, elapsed_time in zip(
            eval_steps,
            training_losses,
            validation_losses,
            eval_times,
        )
    ]

    results = {
        "config": portable_config,
        "history": history,
    }

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(results, file, indent=2)
