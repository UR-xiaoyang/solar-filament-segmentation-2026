from __future__ import annotations

import numpy as np
import pytest

from filament_contract.metrics import (
    DIAGNOSTIC_METRIC_SCOPE,
    diagnostic_instance_metrics,
    diagnostic_semantic_metrics,
)
from filament_contract.postprocess import binary_mask_to_instances


def test_semantic_metrics_are_explicitly_diagnostic() -> None:
    target = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8)

    metrics = diagnostic_semantic_metrics(prediction, target)

    assert metrics.metric_scope == DIAGNOSTIC_METRIC_SCOPE
    assert metrics.intersection == 1
    assert metrics.prediction_area == 2
    assert metrics.target_area == 2
    assert metrics.dice == pytest.approx(0.5)
    assert metrics.iou == pytest.approx(1 / 3)


def test_semantic_empty_mask_policy_is_explicit_and_finite() -> None:
    empty = np.zeros((3, 3), dtype=np.uint8)
    nonempty = empty.copy()
    nonempty[0, 0] = 1

    both_empty = diagnostic_semantic_metrics(empty, empty)
    one_empty = diagnostic_semantic_metrics(nonempty, empty)

    assert (both_empty.dice, both_empty.iou) == (1.0, 1.0)
    assert (one_empty.dice, one_empty.iou) == (0.0, 0.0)


def test_semantic_metrics_validate_shapes_and_binary_values() -> None:
    with pytest.raises(ValueError, match="shapes"):
        diagnostic_semantic_metrics(
            np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 3), dtype=np.uint8)
        )
    with pytest.raises(ValueError, match="binary"):
        diagnostic_semantic_metrics(
            np.array([[2]], dtype=np.uint8), np.array([[1]], dtype=np.uint8)
        )


def test_instance_metrics_require_true_instances_and_report_merge_split() -> None:
    # One prediction joins two targets; another target is split into two predictions.
    target_a = np.zeros((5, 8), dtype=np.uint8)
    target_a[0, 0:2] = 1
    target_b = np.zeros((5, 8), dtype=np.uint8)
    target_b[0, 3:5] = 1
    target_c = np.zeros((5, 8), dtype=np.uint8)
    target_c[3, 1:5] = 1

    merged_prediction = np.zeros((5, 8), dtype=np.uint8)
    merged_prediction[0, 0:5] = 1
    split_prediction_a = np.zeros((5, 8), dtype=np.uint8)
    split_prediction_a[3, 1:3] = 1
    split_prediction_b = np.zeros((5, 8), dtype=np.uint8)
    split_prediction_b[3, 3:5] = 1

    metrics = diagnostic_instance_metrics(
        [merged_prediction, split_prediction_a, split_prediction_b],
        [target_a, target_b, target_c],
        match_iou_threshold=0.35,
    )

    assert metrics.metric_scope == DIAGNOSTIC_METRIC_SCOPE
    assert metrics.matching_policy == "greedy_descending_iou"
    assert metrics.merged_prediction_count == 1
    assert metrics.split_target_count == 1
    assert metrics.target_overlaps_per_prediction == (2, 1, 1)
    assert metrics.prediction_overlaps_per_target == (1, 1, 2)
    assert metrics.matched_count == 2
    assert metrics.false_positive_count == 1
    assert metrics.false_negative_count == 1


def test_instance_metrics_accept_postprocessed_instances() -> None:
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[0, 0:2] = 1
    mask[3, 4] = 1
    predictions = binary_mask_to_instances(mask, connectivity=4)

    metrics = diagnostic_instance_metrics(predictions, [instance.mask for instance in predictions])

    assert metrics.matched_count == 2
    assert metrics.false_positive_count == 0
    assert metrics.false_negative_count == 0
    assert metrics.mean_matched_dice == 1.0
    assert metrics.mean_matched_iou == 1.0


def test_instance_empty_boundaries_do_not_invent_scores() -> None:
    empty = diagnostic_instance_metrics([], [])
    target = np.zeros((2, 2), dtype=np.uint8)
    target[0, 0] = 1
    missed = diagnostic_instance_metrics([], [target])

    assert empty.matched_count == 0
    assert empty.mean_matched_dice is None
    assert empty.mean_matched_iou is None
    assert missed.false_negative_count == 1
    assert missed.absolute_count_error == 1


def test_instance_metrics_reject_empty_instance_and_invalid_threshold() -> None:
    empty_mask = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="foreground"):
        diagnostic_instance_metrics([empty_mask], [])
    with pytest.raises(ValueError, match="match_iou_threshold"):
        diagnostic_instance_metrics([], [], match_iou_threshold=0)
