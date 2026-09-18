import regex as re
import os
import multiprocessing
from .chunking import find_chunk_boundaries

NUM_WORKERS = os.cpu_count() or 1
NUM_CHUNKS = NUM_WORKERS * 2


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
        Given a path to an input text file, the desired vocab size, and special tokens,
        trains a BPE tokenizer returning the vocabulary and the merges produced from training.
    """
    if "<|endoftext|>" in special_tokens:
        special_token = "<|endoftext|>".encode("utf-8")

    elif special_tokens:
        special_token = special_tokens[0].encode("utf-8")

    else:
        special_token = None

    with open(input_path, "rb") as file:  # rb reads file as raw bytes
        if special_token:
            split_boundaries = find_chunk_boundaries(file, NUM_CHUNKS, special_token)

            # Parallel pre-tokenization:
            # 1. Create one argument tuple per chunk.
            # 2. starmap runs pre_tokenize_chunk(...) on those chunks in parallel.
            # 3. Each worker returns a local count dict.
            # 4. We aggregate all local dicts into one global count dict.

            chunks_args = []
            for i in range(len(split_boundaries) - 1):
                chunks_args.append(
                    (
                        input_path,
                        split_boundaries[i],
                        split_boundaries[i + 1],
                        special_tokens
                    )
                )

            with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                individual_pre_token_counts = pool.starmap(pre_tokenize_chunk, chunks_args)

            global_pre_token_count: dict[tuple[bytes, ...], int] = {}
            for curr_pre_token_count in individual_pre_token_counts:
                for token, count in curr_pre_token_count.items():
                    global_pre_token_count[token] = global_pre_token_count.get(token, 0) + count

        else:  # do not use chunking
            global_pre_token_count = pre_tokenize_text(file.read().decode("utf-8"), [])

    vocab = initialize_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    merge_steps = vocab_size - len(vocab)



    # Repeatedly learn the most frequent adjacent byte-pair and merge it
    # until the requested vocabulary size is reached or no pairs remain.
    while merge_steps > 0:
        adjacent_pairs = {}
        for pre_token, count in global_pre_token_count.items():
            for i in range(len(pre_token) - 1):
                sub_token = pre_token[i]
                adjacent_sub_token = pre_token[i + 1]
                adjacent_pairs[(sub_token, adjacent_sub_token)] = adjacent_pairs.get((sub_token, adjacent_sub_token),
                                                                                     0) + count
        if not adjacent_pairs:
            break

        pair_to_merge = max(
            adjacent_pairs,
            # pair with max frequency and if frequencies tie, break lexicographically
            key=lambda pair: (adjacent_pairs[pair], pair)
        )

        merges.append(pair_to_merge)
        merged_token = pair_to_merge[0] + pair_to_merge[1]
        vocab[len(vocab)] = merged_token

        # Apply the selected BPE merge to every pre-token:
        # scan left-to-right, replace each occurrence of the selected pair
        # with the merged token, preserve the original frequency count.
        updated_global_pre_token_count: dict[tuple[bytes, ...], int] = {}
        for pre_token, count in global_pre_token_count.items():
            new_pre_token = []
            i = 0
            while i < len(pre_token):
                if i < len(pre_token) - 1 and pre_token[i] == pair_to_merge[0] and pre_token[i + 1] == pair_to_merge[1]:
                    new_pre_token.append(merged_token)
                    i += 2
                else:
                    new_pre_token.append(pre_token[i])
                    i += 1

            new_pre_token_tuple = tuple(new_pre_token)
            updated_global_pre_token_count[new_pre_token_tuple] = count #every token

        global_pre_token_count = updated_global_pre_token_count
        merge_steps -= 1

    return vocab, merges


def pre_tokenize_chunk(
        input_path: str,
        start: int,
        end: int,
        special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """
    Read and pre-tokenize a byte range from the input file.

    Returns the frequency of each pre-token within the chunk.
    """

    with open(input_path, "rb") as file:  # rb reads file as raw bytes
        file.seek(start)
        chunk = file.read(end - start)
        chunk_string = chunk.decode("utf-8")
        return pre_tokenize_text(chunk_string, special_tokens)


def pre_tokenize_text(
        text: str,
        special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize text while treating special tokens as hard boundaries.

    Splits the input around special tokens, applies the regex pre-tokenizer
    to each remaining segment, and returns pre-token frequencies represented
    as tuples of individual UTF-8 bytes.
    """

    if special_tokens:
        # split around special tokens before pre-tokenization
        special_tokens_split_pattern = "|".join(re.escape(token) for token in special_tokens)
        text_segments = re.split(special_tokens_split_pattern, text)
    else:
        text_segments = [text]

    pre_tokenizer_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_token_count: dict[tuple[bytes, ...], int] = {}
    for segment in text_segments:
        pre_tokens = re.finditer(pre_tokenizer_pattern, segment)
        for match in pre_tokens:
            token = match.group(0)  # e.g. "low"

            # convert the string into one raw bytes object
            token_bytes = token.encode("utf-8")  # e.g. b"low"

            # create tuple of individual bytes
            token_byte_tuple = tuple(bytes([b]) for b in token_bytes)  # e.g. (b"l", b"o", b"w")

            # count how many times we have seen this token across all split segments
            pre_token_count[token_byte_tuple] = pre_token_count.get(token_byte_tuple, 0) + 1

    return pre_token_count


def initialize_vocab(special_tokens: list[str], ) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}  # add all 256 bytes

    for token in special_tokens:  # add special tokens
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab