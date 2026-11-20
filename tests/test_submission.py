from __future__ import annotations

import csv

import numpy as np
import pytest

from filament_contract import (
    IMAGE_SHAPE,
    SOURCE_IMAGE_FORMAT,
    SOURCE_IMAGE_MODE,
    SUBMISSION_COLUMNS,
    SubmissionRecord,
    encode_binary_mask,
    record_from_mask,
    validate_records,
    validate_submission_file,
    write_submission,
)
from filament_contract.submission import MAX_SUBMISSION_FIELD_SIZE


def test_published_source_image_contract_is_explicit() -> None:
    assert IMAGE_SHAPE == (2048, 2048)
    assert SOURCE_IMAGE_FORMAT == "JPEG"
    assert SOURCE_IMAGE_MODE == "L"


@pytest.fixture(scope="module")
def competition_counts() -> str:
    mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    mask[1024, 1024] = 1
    return encode_binary_mask(mask, expected_shape=IMAGE_SHAPE)


def test_record_accepts_arbitrary_suffix_and_requires_image_prefix(
    competition_counts: str,
) -> None:
    valid = SubmissionRecord("image_0042", "image_0042-any-suffix", competition_counts)
    assert validate_records([valid]) == [valid]

    invalid = SubmissionRecord("image_0042", "other-image_0042", competition_counts)
    with pytest.raises(ValueError, match="must start"):
        validate_records([invalid])


def test_filament_id_must_be_unique_across_all_rows(competition_counts: str) -> None:
    first = SubmissionRecord("image_1", "image_1_same", competition_counts)
    second = SubmissionRecord("image_1", "image_1_same", competition_counts)

    with pytest.raises(ValueError, match="duplicate filament_id"):
        validate_records([first, second])


def test_record_from_mask_requires_competition_dimensions() -> None:
    with pytest.raises(ValueError, match="does not match"):
        record_from_mask("image_1", "image_1_a", np.zeros((32, 32), dtype=np.uint8))


def test_writer_emits_only_the_two_contract_fields(tmp_path, competition_counts: str) -> None:
    record = SubmissionRecord("image_1", "image_1_branch-A", competition_counts)
    path = write_submission(tmp_path / "submission.csv", [record])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(SUBMISSION_COLUMNS)
    assert lines[1] == f"{record.filament_id},{competition_counts}"
    assert "size" not in lines[1]
    assert "{" not in lines[1]
    assert '"' not in lines[1]

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "filament_id": record.filament_id,
            "segmentation_rle": competition_counts,
        }
    ]


def test_file_validator_round_trips_writer_output(tmp_path, competition_counts: str) -> None:
    records = [
        SubmissionRecord("img", "img_free", competition_counts),
        SubmissionRecord("img-extended", "img-extended:also-free", competition_counts),
    ]
    path = write_submission(tmp_path / "submission.csv", records)

    assert validate_submission_file(path, {"img", "img-extended"}) == records


def test_file_validator_accepts_valid_rle_larger_than_csv_default(tmp_path) -> None:
    mask = np.random.default_rng(20260713).integers(
        0, 2, size=IMAGE_SHAPE, dtype=np.uint8
    )
    counts = encode_binary_mask(mask, expected_shape=IMAGE_SHAPE)
    assert 131_072 < len(counts) < MAX_SUBMISSION_FIELD_SIZE
    record = SubmissionRecord("large", "large_1", counts)
    path = write_submission(tmp_path / "large.csv", [record])

    assert validate_submission_file(path, {"large"}) == [record]


@pytest.mark.parametrize(
    "header",
    [
        "segmentation_rle,filament_id\n",
        "filament_id\n",
        "filament_id,segmentation_rle,score\n",
    ],
)
def test_file_validator_rejects_wrong_columns(tmp_path, header: str) -> None:
    path = tmp_path / "bad_columns.csv"
    path.write_text(header, encoding="utf-8")

    with pytest.raises(ValueError, match="columns must be exactly"):
        validate_submission_file(path, {"image"})


def test_file_validator_rejects_unknown_prefix(tmp_path, competition_counts: str) -> None:
    path = tmp_path / "unknown.csv"
    path.write_text(
        "filament_id,segmentation_rle\n"
        f"unknown_1,{competition_counts}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no known image ID prefix"):
        validate_submission_file(path, {"known"})


def test_file_validator_rejects_duplicate_ids(tmp_path, competition_counts: str) -> None:
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "filament_id,segmentation_rle\n"
        f"image_1,{competition_counts}\n"
        f"image_1,{competition_counts}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate filament_id"):
        validate_submission_file(path, {"image"})


def test_file_validator_rejects_rle_for_wrong_dimensions(tmp_path) -> None:
    small_counts = encode_binary_mask(np.zeros((8, 8), dtype=np.uint8))
    path = tmp_path / "wrong_dimensions.csv"
    path.write_text(
        "filament_id,segmentation_rle\n" f"image_1,{small_counts}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dimensions"):
        validate_submission_file(path, {"image"})


def test_file_validator_rejects_wrong_field_count(tmp_path, competition_counts: str) -> None:
    path = tmp_path / "extra_field.csv"
    path.write_text(
        "filament_id,segmentation_rle\n"
        f"image_1,{competition_counts},unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wrong field count"):
        validate_submission_file(path, {"image"})
