"""Benchmark tokenizer compression, encoding, and corpus preparation."""

import argparse
import random
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.config import ARTIFACTS_DIR, PROJECT_ROOT
from storylm.tokenization.corpus_encoding import encode_corpus
from storylm.tokenization.tokenizer import Tokenizer


def measure_compression_ratio(tokenizer: Tokenizer) -> None:
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
        token_ids = tokenizer.encode(doc)
        total_num_tokens += len(token_ids)

    print(f"Compression ratio: {total_num_bytes / total_num_tokens:.3f} bytes/token")


def measure_throughput(tokenizer: Tokenizer) -> None:
    """Measure encoding throughput on the TinyStories validation corpus."""
    with (PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt").open(encoding="utf-8") as file:
        text = file.read()

    total_num_bytes = len(text.encode("utf-8"))

    start_time = time.perf_counter()
    tokenizer.encode(text)
    elapsed_time = time.perf_counter() - start_time
    throughput = total_num_bytes / elapsed_time

    print(f"Encoding time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.0f} bytes/second")


def benchmark_corpus_encoding(tokenizer: Tokenizer) -> None:
    """Measure training-corpus encoding and saving to a temporary NumPy file."""
    input_path = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
    file_size_bytes = input_path.stat().st_size

    with TemporaryDirectory(prefix="storylm-benchmark-") as temporary_dir:
        output_path = Path(temporary_dir) / "tokens.npy"
        start_time = time.perf_counter()
        num_tokens = encode_corpus(tokenizer, input_path, output_path)
        elapsed_time = time.perf_counter() - start_time

    print(f"Encoding and saving time: {elapsed_time:.2f} seconds")
    print(f"Tokens: {num_tokens}")
    print(f"End-to-end throughput: {file_size_bytes / elapsed_time:.0f} bytes/second")


def main() -> None:
    """Run the selected tokenizer benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark",
        choices=("compression", "throughput", "corpus"),
        help="Measure compression, validation encoding, or training-corpus encoding and saving.",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        ARTIFACTS_DIR / "tinystories_vocab.pkl",
        ARTIFACTS_DIR / "tinystories_merges.pkl",
        ["<|endoftext|>"],
    )
    benchmarks = {
        "compression": measure_compression_ratio,
        "throughput": measure_throughput,
        "corpus": benchmark_corpus_encoding,
    }
    benchmarks[args.benchmark](tokenizer)


if __name__ == "__main__":
    main()
