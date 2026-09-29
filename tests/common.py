"""Shared fixtures and reference tokenizer utilities."""

from __future__ import annotations

import pathlib
from functools import lru_cache

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Return GPT-2's reversible byte-to-Unicode mapping.

    Printable bytes retain their Unicode representation. The remaining bytes
    are assigned consecutive code points beginning at U+0100. This mapping
    is used to decode the checked-in GPT-2 vocabulary and merge fixtures.
    The mapping algorithm comes from the original GPT-2 implementation.

    Returns:
        A mapping from each byte value to its printable reference character.
    """
    # These 188 bytes retain their printable Unicode representations.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # Assign consecutive code points to the remaining 68 byte values.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # Map a non-printable byte to the next unused Unicode code point.
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
