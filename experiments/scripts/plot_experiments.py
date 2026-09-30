"""Render learning curves from saved experiment histories."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.config import (
    CONTEXT_LENGTH,
    PROJECT_ROOT,
    RESULTS_DIR,
)

FIGURE_DIR = PROJECT_ROOT / "figures"


def load_results(result_path: Path) -> dict:
    """Read a saved experiment configuration and evaluation history."""
    with result_path.open("r") as file:
        return json.load(file)


def plot_learning_curves(
    result_paths: list[Path],
    labels: list[str],
    title: str,
    output_name: str,
    legend_title: str | None = None,
) -> None:
    """Plot validation losses for the supplied experiment records."""
    if len(result_paths) != len(labels):
        raise ValueError("result_paths and labels must have the same length")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for result_path, label in zip(result_paths, labels):
        results = load_results(result_path)

        history = results["history"]

        steps = [entry["step"] for entry in history]
        validation_losses = [entry["validation_loss"] for entry in history]

        plt.plot(
            steps,
            validation_losses,
            label=label,
        )

    plt.xlabel("Training Step")
    plt.ylabel("Per-Token Validation Cross-Entropy Loss")
    plt.title(title)
    plt.legend(title=legend_title)
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
    """Plot the full learning-rate comparisons."""
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
        result_path = RESULTS_DIR / "learning_rate" / experiment_name / "results.json"

        if not result_path.exists():
            print(f"Skipping missing result: {result_path}")
            continue

        result_paths.append(result_path)
        labels.append(label)

    if not result_paths:
        raise FileNotFoundError("No learning-rate experiment results were found.")

    plot_learning_curves(
        result_paths=result_paths,
        labels=labels,
        title="Learning Rate Sweep",
        output_name="learning_rate_validation",
        legend_title="Max Learning Rate",
    )


def plot_learning_rate_stability_probe() -> None:
    """Plot the first 500 steps of the learning-rate stability probes."""
    experiments = [
        (
            RESULTS_DIR / "learning_rate" / "lr_3e-03" / "results.json",
            r"$3 \times 10^{-3}$",
        ),
        (
            RESULTS_DIR / "learning_rate" / "divergence" / "lr_1e-02" / "results.json",
            r"$1 \times 10^{-2}$",
        ),
        (
            RESULTS_DIR / "learning_rate" / "divergence" / "lr_1e-01" / "results.json",
            r"$1 \times 10^{-1}$",
        ),
    ]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for result_path, label in experiments:
        results = load_results(result_path)
        history = results["history"]

        history = [entry for entry in history if entry["step"] <= 500]

        steps = [entry["step"] for entry in history]
        validation_losses = [entry["validation_loss"] for entry in history]

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


def plot_batch_size_sweep() -> None:
    """Plot batch-size comparisons against optimization steps and training tokens."""
    batch_dir = RESULTS_DIR / "batch_size"

    batch_results = []

    if batch_dir.exists():
        for result_path in batch_dir.glob("batch_*/results.json"):
            results = load_results(result_path)
            batch_size = results["config"]["batch_size"]

            batch_results.append((batch_size, result_path))

    # Reuse batch 32 from the optimal LR experiment.
    baseline_path = RESULTS_DIR / "learning_rate" / "lr_3e-03" / "results.json"

    if baseline_path.exists():
        batch_results.append((32, baseline_path))

    if not batch_results:
        print("No batch-size results found.")
        return

    batch_results.sort(key=lambda item: item[0])

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    for batch_size, result_path in batch_results:
        results = load_results(result_path)
        history = results["history"]

        tokens_processed = [entry["step"] * batch_size * CONTEXT_LENGTH for entry in history]

        validation_losses = [entry["validation_loss"] for entry in history]

        plt.plot(
            tokens_processed,
            validation_losses,
            label=f"Batch {batch_size}",
        )

    plt.xlabel("Training Token Predictions")
    plt.ylabel("Per-Token Validation Cross-Entropy Loss")
    plt.title("Batch Size Sweep")
    plt.legend(title="Batch Size")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    pdf_path = FIGURE_DIR / "batch_size_validation.pdf"
    png_path = FIGURE_DIR / "batch_size_validation.png"

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def plot_architecture_ablations() -> None:
    """Plot normalization, positional embedding, and feed-forward comparisons."""
    baseline_path = RESULTS_DIR / "learning_rate" / "lr_3e-03" / "results.json"

    comparisons = [
        (
            baseline_path,
            (RESULTS_DIR / "architecture" / "post_norm" / "results.json"),
            ["Pre-Norm", "Post-Norm"],
            "Pre-Norm vs. Post-Norm",
            "pre_norm_vs_post_norm",
        ),
        (
            baseline_path,
            (RESULTS_DIR / "architecture" / "nope" / "results.json"),
            ["RoPE", "NoPE"],
            "RoPE vs. NoPE",
            "rope_vs_nope",
        ),
        (
            baseline_path,
            (RESULTS_DIR / "architecture" / "silu" / "results.json"),
            ["SwiGLU", "SiLU"],
            "SwiGLU vs. SiLU",
            "swiglu_vs_silu",
        ),
        (
            baseline_path,
            (RESULTS_DIR / "architecture" / "no_rmsnorm" / "lr_1e-03" / "results.json"),
            [
                r"RMSNorm ($3 \times 10^{-3}$)",
                r"No RMSNorm ($1 \times 10^{-3}$)",
            ],
            "RMSNorm vs. No RMSNorm",
            "rmsnorm_ablation",
        ),
    ]

    for (
        baseline,
        ablation,
        labels,
        title,
        output_name,
    ) in comparisons:
        if not baseline.exists():
            print(f"Skipping missing baseline: {baseline}")
            continue

        if not ablation.exists():
            print(f"Skipping missing result: {ablation}")
            continue

        plot_learning_curves(
            result_paths=[
                baseline,
                ablation,
            ],
            labels=labels,
            title=title,
            output_name=output_name,
        )


def plot_no_rmsnorm_learning_rates() -> None:
    """Plot the learning-rate probes for the model without RMSNorm."""
    experiments = [
        (
            RESULTS_DIR / "architecture" / "probes" / "no_rmsnorm_steps_500" / "results.json",
            r"No RMSNorm, LR $3 \times 10^{-3}$",
        ),
        (
            RESULTS_DIR / "architecture" / "probes" / "no_rmsnorm_lr_1e-03_steps_500" / "results.json",
            r"No RMSNorm, LR $1 \times 10^{-3}$",
        ),
    ]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for result_path, label in experiments:
        if not result_path.exists():
            print(f"Skipping missing result: {result_path}")
            continue

        results = load_results(result_path)

        steps = []
        validation_losses = []

        for entry in results["history"]:
            loss = entry["validation_loss"]

            if math.isfinite(loss):
                steps.append(entry["step"])
                validation_losses.append(loss)

        plt.plot(
            steps,
            validation_losses,
            marker="o",
            label=label,
        )

    plt.yscale("log")

    plt.xlabel("Training Step")
    plt.ylabel("Per-Token Validation Cross-Entropy Loss")
    plt.title("No-RMSNorm Learning Rate Stability")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    pdf_path = FIGURE_DIR / "no_rmsnorm_lr_stability.pdf"
    png_path = FIGURE_DIR / "no_rmsnorm_lr_stability.png"

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    plot_learning_rate_sweep()
    plot_learning_rate_stability_probe()
    plot_batch_size_sweep()
    plot_architecture_ablations()
    plot_no_rmsnorm_learning_rates()
