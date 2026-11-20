"""COCO compressed-RLE helpers with strict input validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as mask_utils


Shape = tuple[int, int]


def _normalise_shape(shape: Sequence[int]) -> Shape:
    if isinstance(shape, (str, bytes)) or len(shape) != 2:
        raise ValueError("shape must contain exactly (height, width)")

    height, width = shape
    if (
        isinstance(height, (bool, np.bool_))
        or isinstance(width, (bool, np.bool_))
        or not isinstance(height, (int, np.integer))
        or not isinstance(width, (int, np.integer))
        or height <= 0
        or width <= 0
    ):
        raise ValueError("height and width must be positive integers")
    return int(height), int(width)


def _validate_binary_mask(mask: NDArray[np.generic]) -> None:
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if mask.size == 0:
        raise ValueError("mask must not be empty")
    if mask.dtype.kind not in "buif":
        raise TypeError("mask dtype must be boolean or numeric")
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError("mask must contain only binary values 0 and 1")


def _compressed_counts_to_runs(counts: str) -> list[int]:
    """Decode COCO's counts string enough to validate its run lengths.

    Pixel expansion remains delegated to pycocotools. This small parser prevents
    malformed run totals from reaching the C extension with a claimed shape.
    """

    runs: list[int] = []
    position = 0
    while position < len(counts):
        value = 0
        shift = 0
        more = True
        last_chunk = 0

        while more:
            if position >= len(counts):
                raise ValueError("RLE counts end in an incomplete value")
            code = ord(counts[position])
            position += 1
            if code < 48 or code > 111:
                raise ValueError("RLE counts contain a non-COCO character")

            chunk = code - 48
            value |= (chunk & 0x1F) << (5 * shift)
            more = bool(chunk & 0x20)
            last_chunk = chunk
            shift += 1

        if last_chunk & 0x10:
            value |= -1 << (5 * shift)
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError("RLE counts contain a negative run length")
        runs.append(value)

    if not runs:
        raise ValueError("RLE counts must not be empty")
    return runs


def encode_binary_mask(
    mask: NDArray[np.generic],
    *,
    expected_shape: Sequence[int] | None = None,
) -> str:
    """Encode a 2-D binary mask as pycocotools compressed counts text.

    The returned value is only the ASCII ``counts`` payload. The COCO ``size``
    object is deliberately omitted because the competition CSV does not accept
    it in ``segmentation_rle``.
    """

    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray")
    _validate_binary_mask(mask)

    if expected_shape is not None:
        shape = _normalise_shape(expected_shape)
        if mask.shape != shape:
            raise ValueError(f"mask shape {mask.shape} does not match {shape}")

    fortran_mask = np.asfortranarray(mask, dtype=np.uint8)
    encoded = mask_utils.encode(fortran_mask)
    counts = encoded["counts"]
    if not isinstance(counts, bytes):
        raise TypeError("pycocotools returned non-byte compressed counts")
    return counts.decode("ascii")


def validate_compressed_rle(counts: str, shape: Sequence[int]) -> None:
    """Validate compressed counts against an explicit image shape."""

    height, width = _normalise_shape(shape)
    if not isinstance(counts, str):
        raise TypeError("RLE counts must be an ASCII string")
    try:
        counts_bytes = counts.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("RLE counts must contain ASCII characters only") from exc

    runs = _compressed_counts_to_runs(counts)
    if sum(runs) != height * width:
        raise ValueError(
            "RLE run lengths do not cover the declared image dimensions"
        )

    rle = {"size": [height, width], "counts": counts_bytes}
    try:
        decoded = mask_utils.decode(rle)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("RLE counts are not valid pycocotools compressed RLE") from exc
    if decoded.shape != (height, width):
        raise ValueError("decoded RLE shape does not match the declared dimensions")

    # pycocotools is also the authority on the canonical compressed form.
    if encode_binary_mask(decoded) != counts:
        raise ValueError("RLE counts are not in canonical pycocotools form")


def decode_binary_mask(counts: str, shape: Sequence[int]) -> NDArray[np.uint8]:
    """Decode compressed counts to a C-contiguous uint8 binary mask."""

    normalised_shape = _normalise_shape(shape)
    validate_compressed_rle(counts, normalised_shape)
    rle = {
        "size": list(normalised_shape),
        "counts": counts.encode("ascii"),
    }
    decoded = mask_utils.decode(rle)
    return np.ascontiguousarray(decoded, dtype=np.uint8)
