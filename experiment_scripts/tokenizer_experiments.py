"""Measure tokenizer compression, throughput, and dataset encoding."""

import os
import random
import time

import numpy as np

from storylm.tokenization import tokenizer

from .experiment_config import ARTIFACTS_DIR, PROJECT_ROOT, TRAINING_TOKENS_PATH

tok = tokenizer.Tokenizer.from_files(
    ARTIFACTS_DIR / "tinystories_vocab.pkl", ARTIFACTS_DIR / "tinystories_merges.pkl", ["<|endoftext|>"]
)


def measure_compression_ratio():
    """Estimate bytes per token from ten sampled TinyStories documents."""
    rng = random.Random(67)

    with (PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt").open(encoding="utf-8") as file:
        text = file.read()

    documents = text.split("<|endoftext|>")
    documents = [doc for doc in documents if doc.strip()]
    sampled_documents = rng.sample(documents, 10)

    total_num_bytes = 0
    total_num_tokens = 0
    for doc in sampled_documents:
        total_num_bytes += len(doc.encode("utf-8"))
        token_ids = tok.encode(doc)
        total_num_tokens += len(token_ids)

    print(f"Compression ratio: {total_num_bytes / total_num_tokens:.3f} bytes/token")


def measure_throughput():
    """Measure encoding throughput on the TinyStories validation corpus."""
    with (PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt").open(encoding="utf-8") as file:
        text = file.read()

    total_num_bytes = len(text.encode("utf-8"))

    start_time = time.perf_counter()
    tok.encode(text)
    elapsed_time = time.perf_counter() - start_time
    throughput = total_num_bytes / elapsed_time

    print(f"Encoding time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.0f} bytes/second")


def chunked_encoding():
    """Encode the training corpus in chunks and save uint16 token IDs."""
    input_path = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
    output_path = TRAINING_TOKENS_PATH
    chunk_size = 8 * 1024 * 1024

    # Encode the file incrementally using large text chunks.
    with open(input_path, encoding="utf-8") as file:
        chunks = iter(
            lambda: file.read(chunk_size),
            "",
        )

        start_time = time.perf_counter()

        chunked_token_ids = np.fromiter(
            tok.encode_iterable(chunks),
            dtype=np.uint16,
        )

        elapsed_time = time.perf_counter() - start_time

    print(f"Encoding time: {elapsed_time:.2f} seconds")
    print(f"Tokens: {len(chunked_token_ids)}")
    np.save(output_path, chunked_token_ids)
    print(f"Saved to: {output_path}")

    file_size_bytes = os.path.getsize(input_path)
    throughput = file_size_bytes / elapsed_time
    print(f"Throughput: {throughput:.0f} bytes/second")


if __name__ == "__main__":
    # measure_compression_ratio()
    # measure_throughput()
    chunked_encoding()
