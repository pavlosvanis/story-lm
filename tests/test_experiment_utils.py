"""Privacy and portability checks for saved experiment records."""

import json

from experiments.save_results import save_experiment_results


def test_experiment_results_use_portable_paths(tmp_path):
    """Preserve measurements and inputs while removing local directory names."""
    project_root = tmp_path / "project"
    training_path = project_root / "artifacts" / "train.npy"
    external_path = tmp_path / "private-account" / "checkpoints" / "model.pt"
    config = {
        "training_tokens_path": str(training_path),
        "validation_tokens_path": "artifacts/valid.npy",
        "checkpoint_path": external_path,
        "batch_size": 32,
        "seed": 42,
    }
    original_config = dict(config)
    output_path = project_root / "experiments" / "results.json"

    save_experiment_results(
        out_path=str(output_path),
        config=config,
        training_losses=[2.0, 1.7],
        validation_losses=[2.1, 1.8],
        eval_steps=[100, 200],
        eval_times=[10.0, 20.0],
        project_root=project_root,
    )

    saved_text = output_path.read_text()
    result = json.loads(saved_text)
    assert str(tmp_path) not in saved_text
    assert "private-account" not in saved_text
    assert config == original_config
    assert result["config"] == {
        "training_tokens_path": "artifacts/train.npy",
        "validation_tokens_path": "artifacts/valid.npy",
        "checkpoint_path": "model.pt",
        "batch_size": 32,
        "seed": 42,
    }
    assert result["history"] == [
        {"step": 100, "training_loss": 2.0, "validation_loss": 2.1, "elapsed_seconds": 10.0},
        {"step": 200, "training_loss": 1.7, "validation_loss": 1.8, "elapsed_seconds": 20.0},
    ]
