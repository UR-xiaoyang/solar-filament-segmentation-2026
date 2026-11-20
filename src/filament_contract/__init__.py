"""Public API for the Solar Filament submission contract."""

from .contract import (
    IMAGE_HEIGHT,
    IMAGE_SHAPE,
    IMAGE_WIDTH,
    SOURCE_IMAGE_FORMAT,
    SOURCE_IMAGE_MODE,
    SUBMISSION_COLUMNS,
)
from .rle import decode_binary_mask, encode_binary_mask, validate_compressed_rle
from .submission import (
    SubmissionRecord,
    record_from_mask,
    validate_records,
    validate_submission_file,
    write_submission,
)

__all__ = [
    "IMAGE_HEIGHT",
    "IMAGE_SHAPE",
    "IMAGE_WIDTH",
    "SOURCE_IMAGE_FORMAT",
    "SOURCE_IMAGE_MODE",
    "SUBMISSION_COLUMNS",
    "SubmissionRecord",
    "decode_binary_mask",
    "encode_binary_mask",
    "record_from_mask",
    "validate_compressed_rle",
    "validate_records",
    "validate_submission_file",
    "write_submission",
]
