"""Export inference weights and configuration from the selected checkpoint."""

import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "batch_size" / "batch_64" / "checkpoint.pt"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "final_model"
WEIGHTS_PATH = OUTPUT_DIR / "weights.pt"
CONFIG_PATH = OUTPUT_DIR / "config.json"


MODEL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1344,
    "theta": 10_000,
    "use_rmsnorm": True,
    "norm_style": "pre",
    "use_rope": True,
    "ffn_type": "swiglu",
    "source_experiment": "batch_size_64",
    "final_validation_loss": 1.6303,
}


def main() -> None:
    """Run the command-line entry point."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model_weights = checkpoint["model"]

    torch.save(model_weights, WEIGHTS_PATH)

    with CONFIG_PATH.open("w") as f:
        json.dump(MODEL_CONFIG, f, indent=4)

    print(f"Saved weights to: {WEIGHTS_PATH}")
    print(f"Saved config to:  {CONFIG_PATH}")


if __name__ == "__main__":
    main()
