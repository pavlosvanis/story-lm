"""Offline checks for model reuse, selected training, and inference export."""

import json
import pickle
import random
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from storylm.data.preparation import PreparedData
from storylm.inference.generator import StoryGenerator
from storylm.model.transformer_lm import TransformerLM
from storylm.training import workflow
from storylm.training.checkpointing import save_checkpoint


@pytest.fixture
def tiny_config(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Use a small model while preserving the preparation vocabulary layout."""
    model_config = {
        **workflow.MODEL_CONFIG,
        "context_length": 8,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "d_ff": 16,
    }
    training_config = {
        **workflow.TRAINING_CONFIG,
        "num_iterations": 2,
        "batch_size": 1,
        "eval_batch_size": 1,
        "warmup_iters": 1,
        "cosine_cycle_iters": 1,
        "eval_interval": 1,
        "num_eval_batches": 1,
        "checkpoint_interval": 1,
    }
    monkeypatch.setattr(workflow, "MODEL_CONFIG", model_config)
    monkeypatch.setattr(workflow, "TRAINING_CONFIG", training_config)
    monkeypatch.setattr(workflow, "TOTAL_TOKEN_BUDGET", 16)
    return model_config


def _save_tokenizer(artifacts_dir: Path) -> None:
    """Write a byte vocabulary accepted by real preparation and generation."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    vocab[256] = b"<|endoftext|>"
    vocab.update({token_id: f"unused-{token_id}".encode() for token_id in range(257, 10_000)})
    (artifacts_dir / "tinystories_vocab.pkl").write_bytes(pickle.dumps(vocab))
    (artifacts_dir / "tinystories_merges.pkl").write_bytes(pickle.dumps([]))


def _save_prepared_data(artifacts_dir: Path) -> PreparedData:
    """Create a complete offline preparation cache."""
    _save_tokenizer(artifacts_dir)
    training_path = artifacts_dir / "tinystories_train_tokens.npy"
    validation_path = artifacts_dir / "tinystories_valid_tokens.npy"
    for path in (training_path, validation_path):
        np.save(path, np.arange(16, dtype=np.uint16))
    return PreparedData(
        vocab_path=artifacts_dir / "tinystories_vocab.pkl",
        merges_path=artifacts_dir / "tinystories_merges.pkl",
        training_tokens_path=training_path,
        validation_tokens_path=validation_path,
    )


def _save_model(artifacts_dir: Path, model_config: dict) -> None:
    """Export real small-model weights without requiring training data."""
    model_dir = artifacts_dir / "final_model"
    model_dir.mkdir(parents=True)
    model = TransformerLM(**model_config, device="cpu", dtype=torch.float32)
    torch.save(model.state_dict(), model_dir / "weights.pt")
    (model_dir / "config.json").write_text(json.dumps(model_config), encoding="utf-8")


def _snapshot(directory: Path) -> dict[str, bytes]:
    """Capture artifact contents to detect replacement or extra outputs."""
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def _forbid_new_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if a reuse or invalid-artifact path reaches preparation or training."""
    for name in ("prepare_tinystories", "train_model"):
        monkeypatch.setattr(workflow, name, Mock(side_effect=AssertionError(f"{name} must be skipped")))


def test_training_reuses_model_without_training_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_config: dict
):
    """Load real generation artifacts without preparing data or changing files."""
    artifacts_dir = tmp_path / "artifacts"
    _save_tokenizer(artifacts_dir)
    _save_model(artifacts_dir, tiny_config)
    before = _snapshot(artifacts_dir)
    _forbid_new_training(monkeypatch)
    load = Mock(wraps=workflow.StoryGenerator.from_artifacts)
    monkeypatch.setattr(workflow.StoryGenerator, "from_artifacts", load)

    result = workflow.train_tinystories(tmp_path / "absent-data", artifacts_dir, device="cpu")

    assert result.reused
    assert result.model_dir == artifacts_dir / "final_model"
    load.assert_called_once_with(artifacts_dir, device="cpu")
    assert _snapshot(artifacts_dir) == before
    assert not (tmp_path / "absent-data").exists()


@pytest.mark.parametrize("existing_file", ["weights.pt", "config.json", "checkpoint.pt"])
def test_training_preserves_incomplete_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_file: str):
    """An incomplete model must not trigger preparation, training, or replacement."""
    artifacts_dir = tmp_path / "artifacts"
    model_dir = artifacts_dir / "final_model"
    model_dir.mkdir(parents=True)
    (model_dir / existing_file).write_bytes(b"existing unfinished run")
    before = _snapshot(artifacts_dir)
    _forbid_new_training(monkeypatch)

    with pytest.raises(ValueError, match="incomplete"):
        workflow.train_tinystories(artifacts_dir=artifacts_dir, device="cpu")

    assert _snapshot(artifacts_dir) == before


def test_training_rejects_model_with_missing_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_config: dict
):
    """Weights and config alone must not be mistaken for usable generation artifacts."""
    artifacts_dir = tmp_path / "artifacts"
    _save_model(artifacts_dir, tiny_config)
    before = _snapshot(artifacts_dir)
    _forbid_new_training(monkeypatch)

    with pytest.raises(FileNotFoundError, match="Missing generation artifacts"):
        workflow.train_tinystories(artifacts_dir=artifacts_dir, device="cpu")

    assert _snapshot(artifacts_dir) == before


def test_training_exports_generator_compatible_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_config: dict
):
    """Preserve prepared data and export real loadable weights, history, and a checkpoint."""
    artifacts_dir = tmp_path / "artifacts"
    prepared = _save_prepared_data(artifacts_dir)
    before = _snapshot(artifacts_dir)
    prepare = Mock(wraps=workflow.prepare_tinystories)
    monkeypatch.setattr(workflow, "prepare_tinystories", prepare)
    models = []

    def train(**kwargs):
        assert random.getstate() == random.Random(workflow.SEED).getstate()
        assert np.random.randint(1_000_000) == np.random.RandomState(workflow.SEED).randint(1_000_000)
        assert torch.initial_seed() == workflow.SEED
        assert kwargs["training_tokens_path"] == str(prepared.training_tokens_path)
        assert kwargs["validation_tokens_path"] == str(prepared.validation_tokens_path)
        assert kwargs["device"] == "cpu"
        assert kwargs["dtype"] == torch.float32
        model = TransformerLM(**tiny_config, device="cpu", dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters())
        save_checkpoint(model, optimizer, kwargs["num_iterations"], kwargs["checkpoint_path"])
        models.append(model)
        return model, [2.0, 1.9], [2.1, 2.0], [1, 2], [0.1, 0.2]

    train_mock = Mock(side_effect=train)
    monkeypatch.setattr(workflow, "train_model", train_mock)

    result = workflow.train_tinystories(tmp_path / "absent-data", artifacts_dir, device="cpu")

    assert not result.reused
    assert result.model_dir == artifacts_dir / "final_model"
    prepare.assert_called_once_with(data_dir=tmp_path / "absent-data", artifacts_dir=artifacts_dir)
    train_mock.assert_called_once()
    after = _snapshot(artifacts_dir)
    assert {name: after[name] for name in before} == before
    assert set(after) - set(before) == {
        "final_model/checkpoint.pt",
        "final_model/history.json",
        "final_model/weights.pt",
        "final_model/config.json",
    }
    history = json.loads((result.model_dir / "history.json").read_text(encoding="utf-8"))
    assert history["config"]["total_token_budget"] == 16
    assert history["config"]["seed"] == workflow.SEED
    assert history["config"]["dtype"] == "torch.float32"
    assert history["history"] == [
        {"step": 1, "training_loss": 2.0, "validation_loss": 2.1, "elapsed_seconds": 0.1},
        {"step": 2, "training_loss": 1.9, "validation_loss": 2.0, "elapsed_seconds": 0.2},
    ]
    checkpoint = torch.load(result.model_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["iteration"] == 2
    assert set(checkpoint) == {"model", "optimizer", "iteration"}
    generator = StoryGenerator.from_artifacts(artifacts_dir, device="cpu")
    assert generator.model_config["final_validation_loss"] == 2.0
    assert generator.model_config["source"] == "storylm train"
    for name, expected in models[0].state_dict().items():
        torch.testing.assert_close(generator.model.state_dict()[name], expected)


def test_training_preserves_checkpoint_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_config: dict
):
    """Keep an interrupted run and refuse to start over in the same directory."""
    artifacts_dir = tmp_path / "artifacts"
    _save_prepared_data(artifacts_dir)
    before = _snapshot(artifacts_dir)

    def interrupted_training(**kwargs):
        Path(kwargs["checkpoint_path"]).write_bytes(b"recoverable checkpoint")
        raise RuntimeError("simulated training interruption")

    train = Mock(side_effect=interrupted_training)
    monkeypatch.setattr(workflow, "train_model", train)

    with pytest.raises(RuntimeError, match="simulated training interruption"):
        workflow.train_tinystories(artifacts_dir=artifacts_dir, device="cpu")

    after = _snapshot(artifacts_dir)
    assert {name: after[name] for name in before} == before
    assert after["final_model/checkpoint.pt"] == b"recoverable checkpoint"
    assert not (artifacts_dir / "final_model" / "weights.pt").exists()
    assert not (artifacts_dir / "final_model" / "config.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        workflow.train_tinystories(artifacts_dir=artifacts_dir, device="cpu")
    train.assert_called_once()
    assert _snapshot(artifacts_dir) == after


def test_training_does_not_export_nonfinite_losses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_config: dict):
    """Reject a nonfinite training record before publishing generation artifacts."""
    artifacts_dir = tmp_path / "artifacts"
    _save_prepared_data(artifacts_dir)
    model = TransformerLM(**tiny_config, device="cpu", dtype=torch.float32)
    monkeypatch.setattr(
        workflow,
        "train_model",
        Mock(return_value=(model, [2.0], [float("nan")], [2], [0.1])),
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        workflow.train_tinystories(artifacts_dir=artifacts_dir, device="cpu")

    assert not (artifacts_dir / "final_model" / "weights.pt").exists()
    assert not (artifacts_dir / "final_model" / "config.json").exists()
