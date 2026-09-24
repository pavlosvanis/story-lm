from pathlib import Path
import json

import matplotlib.pyplot as plt

from experiment_scripts.experiment_config import (
    PROJECT_ROOT,
    EXPERIMENTS_DIR,
)

FIGURE_DIR = PROJECT_ROOT / "figures"


def load_results(result_path: Path) -> dict:
    with result_path.open("r") as file:
        return json.load(file)


def plot_learning_curves(
        result_paths: list[Path],
        labels: list[str],
        title: str,
        output_name: str,
) -> None:
    if len(result_paths) != len(labels):
        raise ValueError("result_paths and labels must have the same length")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for result_path, label in zip(result_paths, labels):
        results = load_results(result_path)

        history = results["history"]

        steps = [entry["step"] for entry in history]
        validation_losses = [
            entry["validation_loss"] for entry in history
        ]

        plt.plot(
            steps,
            validation_losses,
            label=label,
        )

    plt.xlabel("Training Step")
    plt.ylabel("Per-Token Validation Cross-Entropy Loss")
    plt.title(title)
    plt.legend(title="Max Learning Rate")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    pdf_path = FIGURE_DIR / f"{output_name}.pdf"
    png_path = FIGURE_DIR / f"{output_name}.png"

    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def plot_learning_rate_sweep() -> None:
    learning_rates = [
        ("lr_1e-04", r"$1 \times 10^{-4}$"),
        ("lr_3e-04", r"$3 \times 10^{-4}$"),
        ("lr_6e-04", r"$6 \times 10^{-4}$"),
        ("lr_1e-03", r"$1 \times 10^{-3}$"),
        ("lr_3e-03", r"$3 \times 10^{-3}$"),
    ]

    result_paths = []
    labels = []

    for experiment_name, label in learning_rates:
        result_path = (
                EXPERIMENTS_DIR
                / "learning_rate"
                / experiment_name
                / "results.json"
        )

        if not result_path.exists():
            print(f"Skipping missing result: {result_path}")
            continue

        result_paths.append(result_path)
        labels.append(label)

    if not result_paths:
        raise FileNotFoundError(
            "No learning-rate experiment results were found."
        )

    plot_learning_curves(
        result_paths=result_paths,
        labels=labels,
        title="Learning Rate Sweep",
        output_name="learning_rate_validation",
    )


def plot_learning_rate_stability_probe() -> None:
    experiments = [
        (
            EXPERIMENTS_DIR
            / "learning_rate"
            / "lr_3e-03"
            / "results.json",
            r"$3 \times 10^{-3}$",
        ),
        (
            EXPERIMENTS_DIR
            / "learning_rate"
            / "divergence"
            / "lr_1e-02"
            / "results.json",
            r"$1 \times 10^{-2}$",
        ),
        (
            EXPERIMENTS_DIR
            / "learning_rate"
            / "divergence"
            / "lr_1e-01"
            / "results.json",
            r"$1 \times 10^{-1}$",
        ),
    ]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for result_path, label in experiments:
        results = load_results(result_path)
        history = results["history"]

        history = [
            entry
            for entry in history
            if entry["step"] <= 500
        ]

        steps = [entry["step"] for entry in history]
        validation_losses = [
            entry["validation_loss"]
            for entry in history
        ]

        plt.plot(
            steps,
            validation_losses,
            marker="o",
            label=label,
        )

    plt.xlabel("Training Step")
    plt.ylabel("Per-Token Validation Cross-Entropy Loss")
    plt.title("Learning Rate Stability Probe: First 500 Steps")
    plt.legend(title="Max Learning Rate")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    pdf_path = FIGURE_DIR / "learning_rate_stability.pdf"
    png_path = FIGURE_DIR / "learning_rate_stability.png"

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    plot_learning_rate_sweep()
    plot_learning_rate_stability_probe()
