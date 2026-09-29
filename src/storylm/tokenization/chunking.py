"""Document-aligned file chunking for parallel pre-tokenization."""

import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find independent file chunks at special-token boundaries.

    Candidate boundaries are advanced to the next occurrence of the supplied
    special token. Overlapping boundaries are deduplicated, so the result may
    contain fewer chunks than requested.

    Args:
        file: Binary input file whose contents will be split.
        desired_num_chunks: Requested number of file chunks.
        split_special_token: Byte sequence marking a valid chunk boundary.

    Returns:
        Sorted byte offsets, including the beginning and end of the file.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Start with uniformly spaced candidate boundaries.
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead in 4 KiB blocks.

    # Advance interior boundaries to the supplied special token so chunks
    # do not split ordinary pre-tokens or UTF-8 characters.
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # Candidates without a later special token collapse to EOF.
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Several candidates may resolve to the same boundary.
    return sorted(set(chunk_boundaries))
