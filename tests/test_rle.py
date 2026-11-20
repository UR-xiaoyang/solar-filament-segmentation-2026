from __future__ import annotations

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from filament_contract import (
    decode_binary_mask,
    encode_binary_mask,
    validate_compressed_rle,
)


def test_round_trip_preserves_binary_mask() -> None:
    rng = np.random.default_rng(2026)
    mask = (rng.random((19, 31)) > 0.82).astype(np.uint8)

    counts = encode_binary_mask(mask)
    decoded = decode_binary_mask(counts, mask.shape)

    assert isinstance(counts, str)
    counts.encode("ascii")
    np.testing.assert_array_equal(decoded, mask)


def test_encoder_delegates_a_fortran_array_to_pycocotools(monkeypatch) -> None:
    original_encode = mask_utils.encode
    observed: dict[str, object] = {}

    def inspecting_encode(mask):
        observed["fortran"] = mask.flags.f_contiguous
        observed["dtype"] = mask.dtype
        return original_encode(mask)

    monkeypatch.setattr("filament_contract.rle.mask_utils.encode", inspecting_encode)
    encode_binary_mask(np.eye(8, dtype=np.uint8))

    assert observed == {"fortran": True, "dtype": np.dtype(np.uint8)}


def test_single_pixel_matches_direct_pycocotools_encoding() -> None:
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[2, 3] = 1

    counts = encode_binary_mask(mask)
    direct = mask_utils.encode(np.asfortranarray(mask))["counts"].decode("ascii")

    assert counts == direct
    decoded = decode_binary_mask(counts, mask.shape)
    assert decoded.sum() == 1
    assert decoded[2, 3] == 1


@pytest.mark.parametrize(
    "mask, error",
    [
        ([[0, 1]], TypeError),
        (np.zeros((2, 2, 1), dtype=np.uint8), ValueError),
        (np.empty((0, 2), dtype=np.uint8), ValueError),
        (np.array([[0, 2]], dtype=np.uint8), ValueError),
        (np.array([[0.0, np.nan]]), ValueError),
        (np.array([["0", "1"]]), TypeError),
    ],
)
def test_encoder_rejects_invalid_masks(mask, error) -> None:
    with pytest.raises(error):
        encode_binary_mask(mask)


def test_encoder_rejects_wrong_expected_shape() -> None:
    with pytest.raises(ValueError, match="does not match"):
        encode_binary_mask(np.zeros((4, 4), dtype=np.uint8), expected_shape=(5, 4))


@pytest.mark.parametrize(
    "counts, error",
    [
        (b"123", TypeError),
        ("", ValueError),
        ("\u592a\u9633", ValueError),
        ("!", ValueError),
        ("P", ValueError),
    ],
)
def test_decoder_rejects_invalid_counts(counts, error) -> None:
    with pytest.raises(error):
        decode_binary_mask(counts, (4, 4))


def test_rle_dimensions_are_part_of_validation_contract() -> None:
    counts = encode_binary_mask(np.zeros((8, 8), dtype=np.uint8))

    validate_compressed_rle(counts, (8, 8))
    with pytest.raises(ValueError, match="dimensions"):
        validate_compressed_rle(counts, (8, 9))

