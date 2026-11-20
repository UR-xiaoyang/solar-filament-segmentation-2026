from __future__ import annotations

import numpy as np
import pytest

from filament_contract.contract import IMAGE_SHAPE
from filament_contract.postprocess import (
    InstanceMask,
    binary_mask_to_instances,
    instances_to_submission_records,
    probability_to_instances,
    threshold_probability_map,
)
from filament_contract.rle import decode_binary_mask


def test_threshold_probability_map_includes_exact_ties() -> None:
    probability = np.array([[0.49, 0.5, 1.0]], dtype=np.float32)

    mask = threshold_probability_map(probability, threshold=0.5)

    np.testing.assert_array_equal(mask, [[0, 1, 1]])
    assert mask.dtype == np.uint8
    assert mask.flags.c_contiguous


@pytest.mark.parametrize(
    "probability, error",
    [
        ([[0.5]], TypeError),
        (np.zeros((1, 1, 1)), ValueError),
        (np.empty((0, 2)), ValueError),
        (np.array([[np.nan]]), ValueError),
        (np.array([[-0.1]]), ValueError),
        (np.array([[1.1]]), ValueError),
        (np.array([["0.5"]]), TypeError),
    ],
)
def test_threshold_probability_map_rejects_invalid_input(probability, error) -> None:
    with pytest.raises(error):
        threshold_probability_map(probability)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.inf])
def test_threshold_probability_map_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        threshold_probability_map(np.zeros((2, 2)), threshold=threshold)


def test_connectivity_changes_diagonal_component_count() -> None:
    mask = np.eye(3, dtype=np.uint8)

    four_connected = binary_mask_to_instances(mask, connectivity=4)
    eight_connected = binary_mask_to_instances(mask, connectivity=8)

    assert [instance.area for instance in four_connected] == [1, 1, 1]
    assert [instance.instance_id for instance in four_connected] == [1, 2, 3]
    assert [instance.area for instance in eight_connected] == [3]


def test_area_filter_score_and_component_order_are_deterministic() -> None:
    probability = np.zeros((6, 8), dtype=np.float32)
    probability[0, 5:7] = [0.6, 0.8]  # area 2, first among equal-area components
    probability[2, 1:3] = [0.9, 0.7]  # area 2
    probability[4, 4:7] = [0.5, 0.6, 0.7]  # area 3, sorted first
    probability[5, 0] = 0.9  # filtered

    instances = probability_to_instances(
        probability, threshold=0.5, min_area=2, connectivity=4
    )

    assert [instance.instance_id for instance in instances] == [1, 2, 3]
    assert [instance.area for instance in instances] == [3, 2, 2]
    assert [instance.score for instance in instances] == pytest.approx([0.6, 0.7, 0.8])
    assert [int(np.flatnonzero(instance.mask)[0]) for instance in instances] == [36, 5, 17]


def test_empty_foreground_produces_no_instances() -> None:
    assert probability_to_instances(np.zeros((4, 5)), threshold=0.5) == []


def test_instance_cap_keeps_largest_components_before_allocating_masks() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2, 0:2] = 1
    mask[4, 0:3] = 1

    instances = binary_mask_to_instances(mask, connectivity=4, max_instances=2)

    assert [instance.area for instance in instances] == [3, 2]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"connectivity": 6}, ValueError),
        ({"connectivity": True}, TypeError),
        ({"min_area": 0}, ValueError),
        ({"min_area": 1.5}, TypeError),
        ({"max_instances": 0}, ValueError),
        ({"max_instances": True}, TypeError),
    ],
)
def test_component_options_are_strict(kwargs, error) -> None:
    with pytest.raises(error):
        binary_mask_to_instances(np.zeros((2, 2), dtype=np.uint8), **kwargs)


def test_submission_records_resort_and_assign_one_based_ids() -> None:
    small = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    small[0, 0] = 1
    large = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    large[100, 100:103] = 1
    instances = [
        InstanceMask(instance_id=9, mask=small, area=1, score=0.9),
        InstanceMask(instance_id=3, mask=large, area=3, score=0.8),
    ]

    records = instances_to_submission_records("test_image", instances)

    assert [record.filament_id for record in records] == ["test_image_1", "test_image_2"]
    np.testing.assert_array_equal(
        decode_binary_mask(records[0].segmentation_rle, IMAGE_SHAPE), large
    )
    np.testing.assert_array_equal(
        decode_binary_mask(records[1].segmentation_rle, IMAGE_SHAPE), small
    )


def test_submission_conversion_rejects_stale_area() -> None:
    mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    mask[0, 0] = 1
    instance = InstanceMask(instance_id=1, mask=mask, area=2, score=1.0)

    with pytest.raises(ValueError, match="area"):
        instances_to_submission_records("test_image", [instance])
