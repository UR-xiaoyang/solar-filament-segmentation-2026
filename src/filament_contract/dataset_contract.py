"""Frozen competition contract constants, errors, and shared primitives."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


COMPETITION_SLUG = "filament-segmentation-2026"
DATASET_DIRECTORY = "MAGFiLO_1.0_Kaggle_2026"
ANNOTATION_FILE = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_NAME_PATTERN = re.compile(r"^(?P<timestamp>\d{14})(?P<site>[A-Za-z]{2})\.jpeg$")

IMAGE_MANIFEST_COLUMNS = (
    "split",
    "image_id",
    "file_name",
    "relative_path",
    "width",
    "height",
    "mode",
    "format",
    "size_bytes",
    "sha256",
    "annotation_record_count",
    "instance_count",
)

ANNOTATION_RECORD_COLUMNS = (
    "image_record_id",
    "image_id",
    "file_name",
    "annotator_batch",
    "date_captured",
    "instance_count",
)


class DatasetContractError(ValueError):
    """Raised when downloaded data violates the frozen competition contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetContractError(message)


def _distribution(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(Counter(values).items(), key=lambda item: item[0])
    }


__all__ = [
    "ANNOTATION_FILE",
    "ANNOTATION_RECORD_COLUMNS",
    "COMPETITION_SLUG",
    "DATASET_DIRECTORY",
    "DatasetContractError",
    "IMAGE_MANIFEST_COLUMNS",
    "IMAGE_NAME_PATTERN",
    "sha256_file",
]
