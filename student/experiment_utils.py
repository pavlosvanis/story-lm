import json
from pathlib import Path


def save_experiment_results(
        out_path: str,
        config: dict,
        training_losses: list[float],
        validation_losses: list[float],
        eval_steps: list[int],
        eval_times: list[float],
) -> None:
    """Save an experiment configuration and evaluation history."""

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
        "config": config,
        "history": history,
    }

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(results, file, indent=2)