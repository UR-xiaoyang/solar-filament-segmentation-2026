"""Construction and validation of competition submission CSV files."""

from __future__ import annotations

import csv
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .contract import IMAGE_SHAPE, SUBMISSION_COLUMNS
from .rle import encode_binary_mask, validate_compressed_rle


MAX_SUBMISSION_FIELD_SIZE = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SubmissionRecord:
    """A validated row plus its image ID, which is not written to the CSV."""

    image_id: str
    filament_id: str
    segmentation_rle: str


def _validate_identifier(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if "\r" in value or "\n" in value:
        raise ValueError(f"{name} must not contain a line break")
    return value


def _validate_record(record: SubmissionRecord) -> None:
    if not isinstance(record, SubmissionRecord):
        raise TypeError("each row must be a SubmissionRecord")
    image_id = _validate_identifier("image_id", record.image_id)
    filament_id = _validate_identifier("filament_id", record.filament_id)
    if not filament_id.startswith(image_id):
        raise ValueError(
            f"filament_id {filament_id!r} must start with image_id {image_id!r}"
        )
    validate_compressed_rle(record.segmentation_rle, IMAGE_SHAPE)


def record_from_mask(
    image_id: str,
    filament_id: str,
    mask: NDArray[np.generic],
) -> SubmissionRecord:
    """Build one submission record from a competition-sized instance mask."""

    record = SubmissionRecord(
        image_id=image_id,
        filament_id=filament_id,
        segmentation_rle=encode_binary_mask(mask, expected_shape=IMAGE_SHAPE),
    )
    _validate_record(record)
    return record


def validate_records(records: Iterable[SubmissionRecord]) -> list[SubmissionRecord]:
    """Materialise and validate records, including row-wide ID uniqueness."""

    materialised = list(records)
    seen_ids: set[str] = set()
    for record in materialised:
        _validate_record(record)
        if record.filament_id in seen_ids:
            raise ValueError(f"duplicate filament_id: {record.filament_id!r}")
        seen_ids.add(record.filament_id)
    return materialised


def write_submission(
    path: str | PathLike[str], records: Iterable[SubmissionRecord]
) -> Path:
    """Write exactly the two published submission columns as UTF-8 CSV."""

    validated = validate_records(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(SUBMISSION_COLUMNS),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in validated:
            writer.writerow(
                {
                    "filament_id": record.filament_id,
                    "segmentation_rle": record.segmentation_rle,
                }
            )
    return output_path


def _match_image_id(filament_id: str, image_ids: Collection[str]) -> str:
    matches = [image_id for image_id in image_ids if filament_id.startswith(image_id)]
    if not matches:
        raise ValueError(f"filament_id {filament_id!r} has no known image ID prefix")
    return max(matches, key=len)


def validate_submission_file(
    path: str | PathLike[str], valid_image_ids: Collection[str]
) -> list[SubmissionRecord]:
    """Read and strictly validate an already-produced competition CSV."""

    image_ids = tuple(valid_image_ids)
    for image_id in image_ids:
        _validate_identifier("image_id", image_id)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("valid_image_ids must not contain duplicates")

    input_path = Path(path)
    records: list[SubmissionRecord] = []
    previous_field_limit = csv.field_size_limit()
    csv.field_size_limit(max(previous_field_limit, MAX_SUBMISSION_FIELD_SIZE))
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
                raise ValueError(
                    f"CSV columns must be exactly {SUBMISSION_COLUMNS!r} in that order"
                )
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"CSV row {line_number} has the wrong field count")
                filament_id = _validate_identifier("filament_id", row["filament_id"])
                image_id = _match_image_id(filament_id, image_ids)
                records.append(
                    SubmissionRecord(
                        image_id=image_id,
                        filament_id=filament_id,
                        segmentation_rle=row["segmentation_rle"],
                    )
                )
    finally:
        csv.field_size_limit(previous_field_limit)
    return validate_records(records)
