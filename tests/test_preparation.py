"""Offline checks for TinyStories preparation and artifact reuse."""

import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from storylm.data import preparation


def _vocabulary() -> dict[int, bytes]:
    """Build a test vocabulary with byte tokens and unused padding entries."""
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    vocab[256] = b"<|endoftext|>"
    vocab.update({token_id: f"unused-{token_id}".encode() for token_id in range(257, 10_000)})
    return vocab


def _save_tokenizer(artifacts_dir: Path) -> None:
    """Create tokenizer files in the test's temporary artifact directory."""
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "tinystories_vocab.pkl").write_bytes(pickle.dumps(_vocabulary()))
    (artifacts_dir / "tinystories_merges.pkl").write_bytes(pickle.dumps([]))


def _snapshot(directory: Path) -> dict[str, bytes]:
    """Capture file contents to detect changes to existing artifacts."""
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_prepare_reuses_all_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A complete cache must avoid downloading, training, encoding, and writes."""
    artifacts_dir = tmp_path / "artifacts"
    _save_tokenizer(artifacts_dir)
    for name in ("tinystories_train_tokens.npy", "tinystories_valid_tokens.npy"):
        np.save(artifacts_dir / name, np.arange(8, dtype=np.uint16))
    weights_path = artifacts_dir / "final_model" / "weights.pt"
    weights_path.parent.mkdir()
    weights_path.write_bytes(b"existing model weights")
    before = _snapshot(artifacts_dir)

    for name in ("download_tinystories_split", "train_bpe", "encode_corpus"):
        monkeypatch.setattr(preparation, name, Mock(side_effect=AssertionError(f"{name} must be skipped")))

    prepared = preparation.prepare_tinystories(tmp_path / "absent-data", artifacts_dir)

    assert prepared.training_tokens_path == artifacts_dir / "tinystories_train_tokens.npy"
    assert prepared.validation_tokens_path == artifacts_dir / "tinystories_valid_tokens.npy"
    assert _snapshot(artifacts_dir) == before
    assert not (tmp_path / "absent-data").exists()


def test_prepare_encodes_only_missing_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Prepare missing validation tokens while preserving the existing cache."""
    artifacts_dir = tmp_path / "artifacts"
    _save_tokenizer(artifacts_dir)
    np.save(artifacts_dir / "tinystories_train_tokens.npy", np.arange(8, dtype=np.uint16))
    before = _snapshot(artifacts_dir)
    validation_text = tmp_path / "validation.txt"
    validation_text.write_text("B owl.<|endoftext|>", encoding="utf-8")
    download = Mock(return_value=validation_text)
    monkeypatch.setattr(preparation, "download_tinystories_split", download)
    monkeypatch.setattr(preparation, "train_bpe", Mock(side_effect=AssertionError("BPE training must be skipped")))

    prepared = preparation.prepare_tinystories(tmp_path / "data", artifacts_dir)

    download.assert_called_once_with("validation", tmp_path / "data")
    np.testing.assert_array_equal(
        np.load(prepared.validation_tokens_path),
        np.array([66, 32, 111, 119, 108, 46, 256], dtype=np.uint16),
    )
    after = _snapshot(artifacts_dir)
    assert {name: after[name] for name in before} == before
    assert set(after) - set(before) == {"tinystories_valid_tokens.npy"}


def test_prepare_from_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a tokenizer and both token arrays without network or expensive BPE."""
    raw_dir = tmp_path / "data"
    raw_dir.mkdir()
    training_text = raw_dir / "train.txt"
    validation_text = raw_dir / "validation.txt"
    training_text.write_text("A fox.<|endoftext|>", encoding="utf-8")
    validation_text.write_text("B owl.<|endoftext|>", encoding="utf-8")
    downloads = []

    def download(split, data_dir):
        downloads.append((split, data_dir))
        return {"train": training_text, "validation": validation_text}[split]

    train = Mock(return_value=(_vocabulary(), []))
    monkeypatch.setattr(preparation, "download_tinystories_split", download)
    monkeypatch.setattr(preparation, "train_bpe", train)

    prepared = preparation.prepare_tinystories(raw_dir, tmp_path / "artifacts")

    train.assert_called_once_with(
        input_path=str(training_text),
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    assert downloads == [("train", raw_dir), ("validation", raw_dir)]
    assert prepared.vocab_path.is_file()
    assert prepared.merges_path.is_file()
    for path, expected in (
        (prepared.training_tokens_path, [65, 32, 102, 111, 120, 46, 256]),
        (prepared.validation_tokens_path, [66, 32, 111, 119, 108, 46, 256]),
    ):
        actual = np.load(path)
        assert actual.dtype == np.uint16
        np.testing.assert_array_equal(actual, expected)


def test_prepare_does_not_cache_failed_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An interrupted encoding must preserve the cache and leave no final output."""
    artifacts_dir = tmp_path / "artifacts"
    _save_tokenizer(artifacts_dir)
    np.save(artifacts_dir / "tinystories_train_tokens.npy", np.arange(8, dtype=np.uint16))
    before = _snapshot(artifacts_dir)
    validation_text = tmp_path / "validation.txt"
    validation_text.write_text("B owl.<|endoftext|>", encoding="utf-8")
    monkeypatch.setattr(preparation, "download_tinystories_split", Mock(return_value=validation_text))

    def interrupted_encoding(tokenizer, input_path, output_path):
        output_path.write_bytes(b"partial NumPy file")
        raise OSError("simulated write failure")

    monkeypatch.setattr(preparation, "encode_corpus", interrupted_encoding)

    with pytest.raises(OSError, match="simulated write failure"):
        preparation.prepare_tinystories(tmp_path / "data", artifacts_dir)

    assert _snapshot(artifacts_dir) == before
    assert not (artifacts_dir / "tinystories_valid_tokens.npy").exists()
    assert all(path.is_file() for path in artifacts_dir.iterdir())


@pytest.mark.parametrize("filename", ["tinystories_vocab.pkl", "tinystories_merges.pkl"])
def test_prepare_rejects_incomplete_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str):
    """A single tokenizer file must not trigger replacement training."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / filename).write_bytes(b"existing partial tokenizer")
    before = _snapshot(artifacts_dir)
    monkeypatch.setattr(
        preparation,
        "download_tinystories_split",
        Mock(side_effect=AssertionError("Downloading must be skipped")),
    )

    with pytest.raises(ValueError, match="incomplete"):
        preparation.prepare_tinystories(tmp_path / "data", artifacts_dir)

    assert _snapshot(artifacts_dir) == before


def test_prepare_preserves_model_with_missing_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Existing model weights must not be paired with a replacement tokenizer."""
    artifacts_dir = tmp_path / "artifacts"
    weights_path = artifacts_dir / "final_model" / "weights.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"existing model weights")
    before = _snapshot(artifacts_dir)
    monkeypatch.setattr(
        preparation,
        "download_tinystories_split",
        Mock(side_effect=AssertionError("Downloading must be skipped")),
    )

    with pytest.raises(ValueError, match="Tokenizer files are missing"):
        preparation.prepare_tinystories(tmp_path / "data", artifacts_dir)

    assert _snapshot(artifacts_dir) == before
