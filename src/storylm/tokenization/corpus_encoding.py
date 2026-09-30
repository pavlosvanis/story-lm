"""Encode text corpora using the existing streaming BPE tokenizer."""

from pathlib import Path

import numpy as np

from .tokenizer import Tokenizer


def encode_corpus(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """Encode a text corpus in chunks and save uint16 token IDs.

    Text is read in chunks; the complete token array is collected before saving.

    Args:
        tokenizer: Tokenizer used to encode the corpus.
        input_path: UTF-8 text file to encode.
        output_path: Destination NumPy file. Its parent directory must exist.

    Returns:
        Number of token IDs written.
    """
    chunk_size = 8 * 1024 * 1024

    # Encode the file incrementally using large text chunks.
    with open(input_path, encoding="utf-8") as file:
        chunks = iter(
            lambda: file.read(chunk_size),
            "",
        )

        chunked_token_ids = np.fromiter(
            tokenizer.encode_iterable(chunks),
            dtype=np.uint16,
        )

    np.save(output_path, chunked_token_ids)
    return len(chunked_token_ids)
