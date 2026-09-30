"""Download and verify the TinyStories GPT-4 V2 training and validation files."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from urllib.request import Request, urlopen

from tqdm import tqdm

# Pin the source revision so future dataset updates do not change training input.
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATASET_URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{DATASET_REVISION}"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DatasetFile:
    """Expected filename, byte count, and SHA-256 checksum for a dataset split."""

    filename: str
    size_bytes: int
    sha256: str


DATASET_FILES = {
    "train": DatasetFile(
        filename="TinyStoriesV2-GPT4-train.txt",
        size_bytes=2_227_753_162,
        sha256="6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443",
    ),
    "validation": DatasetFile(
        filename="TinyStoriesV2-GPT4-valid.txt",
        size_bytes=22_502_601,
        sha256="6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d",
    ),
}


def download_tinystories_split(
    split: Literal["train", "validation"],
    data_dir: str | Path = "data",
) -> Path:
    """Return a verified local dataset file, downloading it if missing.

    Existing files are checked before reuse. An invalid existing file raises an
    error rather than being overwritten. Downloads use a temporary directory on
    the same filesystem and are moved into place only after verification.

    Args:
        split: Dataset split to prepare.
        data_dir: Destination directory, relative to the current working
            directory unless an absolute path is supplied.

    Returns:
        Absolute path to the verified dataset file.

    Raises:
        ValueError: If the split is unknown or a file fails verification.
        OSError: If downloading or reading/writing the file fails.
    """
    if split not in DATASET_FILES:
        raise ValueError(f"Unknown TinyStories split: {split!r}")

    expected = DATASET_FILES[split]
    data_dir = Path(data_dir).expanduser().resolve()
    destination = data_dir / expected.filename

    if destination.exists():
        print(f"Verifying existing dataset: {destination.name}")
        _verify_file(destination, expected)
        return destination

    data_dir.mkdir(parents=True, exist_ok=True)
    request = Request(
        f"{DATASET_URL}/{expected.filename}",
        headers={"User-Agent": "StoryLM"},
    )

    # Keep partial downloads separate from the final filename.
    with TemporaryDirectory(prefix=".storylm-download-", dir=data_dir) as temporary_dir:
        temporary_path = Path(temporary_dir) / expected.filename
        digest = hashlib.sha256()

        with (
            urlopen(request, timeout=60) as response,
            temporary_path.open("wb") as output,
            tqdm(
                total=expected.size_bytes,
                desc=expected.filename,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress,
        ):
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                progress.update(len(chunk))

        _verify_file(temporary_path, expected, checksum=digest.hexdigest())
        temporary_path.replace(destination)

    return destination


def _verify_file(
    path: Path,
    expected: DatasetFile,
    *,
    checksum: str | None = None,
) -> None:
    """Check file size and checksum, hashing the file if no checksum is supplied."""
    actual_size = path.stat().st_size
    if actual_size != expected.size_bytes:
        raise ValueError(
            f"Dataset size mismatch for {path.name}: expected {expected.size_bytes} bytes, found {actual_size}."
        )

    if checksum is None:
        with path.open("rb") as file:
            checksum = hashlib.file_digest(file, "sha256").hexdigest()

    if checksum != expected.sha256:
        raise ValueError(f"Dataset checksum mismatch for {path.name}; the file does not match the pinned source.")
