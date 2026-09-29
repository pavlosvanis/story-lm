"""Train and serialize the TinyStories byte-level BPE tokenizer."""

import pickle
import time

from storylm.tokenization.bpe_training import train_bpe

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"

    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed_time = time.perf_counter() - start_time

    longest_token = max(vocab.values(), key=len)

    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Longest token: {longest_token!r}")
    print(f"Longest token length: {len(longest_token)} bytes")

    with open("artifacts/tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("artifacts/tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    print("Saved vocabulary to tinystories_vocab.pkl")
    print("Saved merges to tinystories_merges.pkl")
