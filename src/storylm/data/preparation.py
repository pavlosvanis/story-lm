"""Prepare TinyStories tokenizer artifacts and tokenized training corpora."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from storylm.tokenization.bpe_training import train_bpe
from storylm.tokenization.corpus_encoding import encode_corpus
from storylm.tokenization.tokenizer import Tokenizer

from .dataset_download import download_tinystories_split

VOCAB_SIZE = 10_000
END_OF_TEXT = "<|endoftext|>"


@dataclass(frozen=True)
class PreparedData:
    """Paths to the tokenizer and token arrays required for model training."""

    vocab_path: Path
    merges_path: Path
    training_tokens_path: Path
    validation_tokens_path: Path


def prepare_tinystories(
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
) -> PreparedData:
    """Prepare missing TinyStories artifacts and reuse existing files.

    Use the existing 10,000-token BPE training and uint16 corpus encoding
    implementations. Existing token arrays are checked for shape and dtype;
    they are assumed to belong to the existing tokenizer.

    Args:
        data_dir: Directory for downloaded raw text files.
        artifacts_dir: Directory for tokenizer files and token arrays.
            Relative directories are resolved from the current working directory.

    Returns:
        Absolute paths to the prepared tokenizer files and token arrays.

    Raises:
        ValueError: If existing artifacts are incomplete or invalid.
        OSError: If downloading or reading/writing files fails.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    artifacts_dir = Path(artifacts_dir).expanduser().resolve()
    prepared = PreparedData(
        vocab_path=artifacts_dir / "tinystories_vocab.pkl",
        merges_path=artifacts_dir / "tinystories_merges.pkl",
        training_tokens_path=artifacts_dir / "tinystories_train_tokens.npy",
        validation_tokens_path=artifacts_dir / "tinystories_valid_tokens.npy",
    )

    vocab_exists = prepared.vocab_path.exists()
    merges_exist = prepared.merges_path.exists()
    if vocab_exists != merges_exist:
        raise ValueError("Tokenizer artifacts are incomplete; both vocabulary and merges files are required.")

    token_paths = (prepared.training_tokens_path, prepared.validation_tokens_path)
    if not vocab_exists and any(path.exists() for path in (*token_paths, artifacts_dir / "final_model" / "weights.pt")):
        raise ValueError(
            "Tokenizer files are missing, but token arrays or model weights exist. "
            "Restore the matching tokenizer or prepare into a separate artifacts directory."
        )

    for path in token_paths:
        if path.exists():
            _validate_token_array(path)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: dict[str, Path] = {}
    if vocab_exists:
        tokenizer = _load_tokenizer(prepared.vocab_path, prepared.merges_path)
        print("Reusing existing tokenizer.")
    else:
        raw_paths["train"] = download_tinystories_split("train", data_dir)
        print("Training the TinyStories BPE tokenizer.")
        vocab, merges = train_bpe(
            input_path=str(raw_paths["train"]),
            vocab_size=VOCAB_SIZE,
            special_tokens=[END_OF_TEXT],
        )

        with TemporaryDirectory(prefix=".storylm-prepare-", dir=artifacts_dir) as temporary_dir:
            temporary_vocab = Path(temporary_dir) / prepared.vocab_path.name
            temporary_merges = Path(temporary_dir) / prepared.merges_path.name
            with temporary_vocab.open("wb") as file:
                pickle.dump(vocab, file)
            with temporary_merges.open("wb") as file:
                pickle.dump(merges, file)
            tokenizer = _load_tokenizer(temporary_vocab, temporary_merges)
            temporary_vocab.replace(prepared.vocab_path)
            temporary_merges.replace(prepared.merges_path)
        print("Saved tokenizer vocabulary and merges.")

    for split, output_path in (
        ("train", prepared.training_tokens_path),
        ("validation", prepared.validation_tokens_path),
    ):
        if output_path.exists():
            print(f"Reusing token array: {output_path.name}")
            continue

        if split not in raw_paths:
            raw_paths[split] = download_tinystories_split(split, data_dir)
        input_path = raw_paths[split]
        print(f"Encoding the {split} corpus.")
        with TemporaryDirectory(prefix=".storylm-prepare-", dir=artifacts_dir) as temporary_dir:
            temporary_output = Path(temporary_dir) / output_path.name
            encode_corpus(tokenizer, input_path, temporary_output)
            _validate_token_array(temporary_output)
            temporary_output.replace(output_path)
        print(f"Saved token array: {output_path.name}")

    return prepared


def _load_tokenizer(vocab_path: Path, merges_path: Path) -> Tokenizer:
    """Load a tokenizer and check the expected TinyStories vocabulary layout."""
    tokenizer = Tokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens=[END_OF_TEXT],
    )
    if (
        len(tokenizer.vocab) != VOCAB_SIZE
        or set(tokenizer.vocab) != set(range(VOCAB_SIZE))
        or tokenizer.vocab.get(256) != END_OF_TEXT.encode("utf-8")
    ):
        raise ValueError("Tokenizer must have 10,000 contiguous token IDs and the expected end-of-text token.")
    return tokenizer


def _validate_token_array(path: Path) -> None:
    """Check a saved token array without reading its complete contents into RAM."""
    tokens = np.load(path, mmap_mode="r", allow_pickle=False)
    if tokens.ndim != 1 or tokens.dtype != np.uint16 or tokens.size == 0:
        raise ValueError(f"Expected a nonempty, one-dimensional uint16 token array: {path.name}")
