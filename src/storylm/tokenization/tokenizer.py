"""Byte-level BPE encoding, streaming tokenization, and decoding."""

import pickle
from collections.abc import Iterable, Iterator
from typing import Self

import regex as re

PRE_TOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """Byte-level BPE tokenizer.

    Encode text using a learned vocabulary and ordered BPE merge rules,
    while preserving configured special tokens as indivisible tokens.
    """

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """Initialize the tokenizer.

        Args:
            vocab: Mapping from token IDs to token byte sequences.
            merges: BPE merge rules in the order they were learned.
            special_tokens: Tokens that should be preserved without pre-tokenization
                or BPE merging.
        """
        self.vocab = vocab  # token_id -> token_bytes

        self.merges = merges  # ordered list of merge pairs
        self.merge_rank = {merges[i]: i for i in range(len(merges))}  # pair -> order_learned
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens_set = set(self.special_tokens)

        # Sort longest-first so that when one special token is a prefix of another,
        # the regex matches the full longer special token instead of prematurely matching the shorter prefix.
        self.special_tokens_for_regex = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        if self.special_tokens_for_regex:
            self.special_tokens_pattern = (
                "("
                + "|".join(
                    re.escape(token) for token in self.special_tokens_for_regex
                )  # need to keep the special tokens
                + ")"
            )
        else:
            self.special_tokens_pattern = None

        for token in self.special_tokens:
            encoded_token = token.encode("utf-8")
            if encoded_token not in self.vocab.values():
                self.vocab[len(self.vocab)] = encoded_token

        self.inverse_vocab = {
            token_bytes: token_id for token_id, token_bytes in self.vocab.items()
        }  # token_bytes -> token_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        """Create a tokenizer from serialized vocabulary and merge files.

        Args:
            vocab_filepath: Path to the serialized vocabulary.
            merges_filepath: Path to the serialized BPE merge rules.
            special_tokens: Tokens that should be preserved during tokenization.

        Returns:
            A tokenizer initialized from the serialized files.
        """
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)

        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using byte-level BPE.

        Special tokens are preserved as single tokens. All other text is
        pre-tokenized before applying the learned BPE merges.

        Args:
            text: Text to encode.

        Returns:
            Token IDs representing the encoded text.
        """
        if self.special_tokens_pattern:
            text_segments = re.split(self.special_tokens_pattern, text)
        else:
            text_segments = [text]

        token_ids = []

        for segment in text_segments:
            if segment not in self.special_tokens_set:
                pre_token_matches = re.finditer(PRE_TOKENIZER_PATTERN, segment)
                for match in pre_token_matches:
                    pre_token_text = match.group(0)  # e.g. "low"

                    # Represent the pre-token as UTF-8 bytes.
                    pre_token_bytes = pre_token_text.encode("utf-8")  # e.g. b"low"

                    # Merges replace adjacent entries in this byte-token list.
                    bpe_tokens = [bytes([b]) for b in pre_token_bytes]  # e.g. [b"l", b"o", b"w"]

                    while True:
                        merge_candidates = []
                        for i in range(len(bpe_tokens) - 1):
                            pair = bpe_tokens[i], bpe_tokens[i + 1]
                            if pair in self.merge_rank:
                                merge_candidates.append(pair)

                        if not merge_candidates:  # no more merges for pre_token_text
                            break
                        updated_bpe_tokens = []
                        pair_to_merge = min(merge_candidates, key=lambda x: self.merge_rank[x])
                        i = 0
                        while i < len(bpe_tokens):
                            current_pair = None
                            if i < len(bpe_tokens) - 1:
                                current_pair = bpe_tokens[i], bpe_tokens[i + 1]

                            if current_pair == pair_to_merge:
                                updated_bpe_tokens.append(pair_to_merge[0] + pair_to_merge[1])
                                i += 2
                            else:
                                updated_bpe_tokens.append(bpe_tokens[i])
                                i += 1

                        bpe_tokens = updated_bpe_tokens

                    for bpe_token in bpe_tokens:
                        token_id = self.inverse_vocab[bpe_token]
                        token_ids.append(token_id)
            else:
                token_ids.append(self.inverse_vocab[segment.encode("utf-8")])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of text chunks into token IDs.

        Buffers incomplete pre-tokens and partial special tokens across chunk
        boundaries so streaming tokenization is consistent with encoding the
        complete text at once.

        Args:
            iterable: Text chunks to encode.

        Yields:
            Token IDs in encoded order.
        """
        # Holds text from the previous chunk that could not yet be safely tokenized.
        buffer = ""

        for chunk in iterable:
            # Prepend any unfinished text from the previous iteration.
            buffer += chunk

            # A chunk may end in the middle of a special token.
            # Find the longest suffix of the buffer that could be the beginning
            # of one of our special tokens, and keep it for the next chunk.
            partial_special_suffix = ""

            for special_token in self.special_tokens:
                for prefix_length in range(1, len(special_token)):
                    prefix = special_token[:prefix_length]

                    if buffer.endswith(prefix) and len(prefix) > len(partial_special_suffix):
                        partial_special_suffix = prefix

            # Only tokenize text that cannot be part of an unfinished special token.
            if partial_special_suffix:
                safe_text = buffer[: -len(partial_special_suffix)]
            else:
                safe_text = buffer

            # Split around complete special tokens while preserving them.
            # Special tokens are hard tokenization boundaries.
            if self.special_tokens_pattern:
                safe_text_segments = re.split(self.special_tokens_pattern, safe_text)
            else:
                safe_text_segments = [safe_text]

            # Every segment except the last is complete because a known boundary
            # follows it, so it can be encoded immediately.
            for segment in safe_text_segments[:-1]:
                yield from self.encode(segment)

            # The final ordinary segment may continue in the next chunk.
            # Its final pre-token therefore cannot yet be safely emitted.
            last_segment = safe_text_segments[-1]
            pre_token_matches = list(re.finditer(PRE_TOKENIZER_PATTERN, last_segment))

            # No ordinary pre-token to retain. Keep only a possible unfinished
            # special-token prefix for the next iteration.
            if not pre_token_matches:
                buffer = partial_special_suffix
                continue

            # All pre-tokens except the final one are known to be complete.
            for match in pre_token_matches[:-1]:
                yield from self.encode(match.group(0))

            # Carry the potentially unfinished final pre-token, together with any
            # unfinished special-token prefix, into the next chunk.
            buffer = pre_token_matches[-1].group(0) + partial_special_suffix

        # End of input means the remaining buffered text can no longer continue,
        # so it is now safe to encode.
        yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text.

        Invalid UTF-8 byte sequences are replaced with the Unicode replacement
        character.

        Args:
            ids: Token IDs to decode.

        Returns:
            The decoded text.
        """
        byte_sequence = []
        for token_id in ids:
            token_bytes = self.vocab[token_id]
            byte_sequence.append(token_bytes)
        decoded_sequence = b"".join(byte_sequence).decode("utf-8", errors="replace")
        return decoded_sequence
