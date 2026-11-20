"""Tests for A5-4 WP0 PQ measurement contract."""

import numpy as np
import pytest

from filament_contract.a5_pq_contract import (
    PQ_CONTRACT_VERSION,
    PQ_THRESHOLD,
    PQ_MIN_AREA,
    PQ_MATCH_IOU,
    CONTRACT_SCOPE,
    compute_pq_metrics,
    aggregate_pq_metrics,
)


def test_pq_contract_constants_frozen():
    """Verify frozen contract parameters match A5-4 WP0 specification."""
    assert PQ_CONTRACT_VERSION == 1
    assert PQ_THRESHOLD == 0.50
    assert PQ_MIN_AREA == 256
    assert PQ_MATCH_IOU == 0.10
    assert CONTRACT_SCOPE == "development_oof_instance_proxy_not_official"


def test_compute_pq_perfect_match():
    """Perfect prediction should yield zero structural error."""
    # Single instance: probability blob above threshold
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[20:40, 30:50] = 0.8  # 400 pixels > min_area=256

    # Ground truth matches exactly
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[20:40, 30:50] = 1

    metrics = compute_pq_metrics(prob, [gt_mask], "test_001")

    assert metrics.image_id == "test_001"
    assert metrics.prediction_count == 1
    assert metrics.target_count == 1
    assert metrics.matched_count == 1
    assert metrics.false_positive_count == 0
    assert metrics.false_negative_count == 0
    assert metrics.split_count == 0
    assert metrics.merge_count == 0
    assert metrics.structural_error == 0.0
    assert metrics.matched_dice_mean == pytest.approx(1.0, abs=0.01)


def test_compute_pq_false_positive():
    """Spurious prediction should count as FP."""
    # Prediction with no ground truth
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[20:40, 30:50] = 0.9

    metrics = compute_pq_metrics(prob, [], "test_fp")

    assert metrics.prediction_count == 1
    assert metrics.target_count == 0
    assert metrics.matched_count == 0
    assert metrics.false_positive_count == 1
    assert metrics.false_negative_count == 0
    assert metrics.structural_error == 1.0  # FP only


def test_compute_pq_false_negative():
    """Missed ground truth should count as FN."""
    # No prediction above threshold
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[20:40, 30:50] = 0.3  # Below PQ_THRESHOLD=0.50

    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[20:40, 30:50] = 1

    metrics = compute_pq_metrics(prob, [gt_mask], "test_fn")

    assert metrics.prediction_count == 0
    assert metrics.target_count == 1
    assert metrics.matched_count == 0
    assert metrics.false_positive_count == 0
    assert metrics.false_negative_count == 1
    assert metrics.structural_error == 1.0  # FN only


def test_compute_pq_split_detection():
    """One ground truth matched by multiple predictions should count split."""
    prob = np.zeros((100, 100), dtype=np.float32)
    # Two separate blobs that together cover one ground truth
    prob[20:35, 30:50] = 0.9  # Fragment 1 (300 pixels)
    prob[36:50, 30:50] = 0.9  # Fragment 2 (280 pixels)

    # Single ground truth covering both
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[20:50, 30:50] = 1

    metrics = compute_pq_metrics(prob, [gt_mask], "test_split")

    assert metrics.prediction_count == 2
    assert metrics.target_count == 1
    assert metrics.matched_count == 1  # Best match wins
    assert metrics.split_count == 1  # One target has multiple overlapping predictions
    assert metrics.structural_error == pytest.approx(1.5)  # 1 FP + 0 FN + 0.5*split


def test_compute_pq_merge_detection():
    """Multiple ground truths matched by one prediction should count merge."""
    # Single large prediction
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[20:50, 30:70] = 0.9  # 1200 pixels

    # Two separate ground truth instances
    gt1 = np.zeros((100, 100), dtype=np.uint8)
    gt1[20:35, 30:50] = 1
    gt2 = np.zeros((100, 100), dtype=np.uint8)
    gt2[36:50, 51:70] = 1

    metrics = compute_pq_metrics(prob, [gt1, gt2], "test_merge")

    assert metrics.prediction_count == 1
    assert metrics.target_count == 2
    assert metrics.matched_count == 1  # Best match wins
    assert metrics.merge_count == 1  # One prediction overlaps multiple targets
    assert metrics.structural_error == pytest.approx(1.5)  # 0 FP + 1 FN + 0.5*merge


def test_compute_pq_below_min_area_filtered():
    """Predictions below min_area=256 should be filtered out."""
    prob = np.zeros((100, 100), dtype=np.float32)
    # Small blob: 10×10 = 100 pixels < 256
    prob[20:30, 30:40] = 0.9

    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[20:30, 30:40] = 1

    metrics = compute_pq_metrics(prob, [gt_mask], "test_small")

    # Prediction filtered → counts as FN
    assert metrics.prediction_count == 0
    assert metrics.target_count == 1
    assert metrics.false_negative_count == 1


def test_compute_pq_low_iou_not_matched():
    """Pairs below match_iou=0.10 should not match."""
    prob = np.zeros((100, 100), dtype=np.float32)
    # Prediction with minimal overlap
    prob[20:40, 30:50] = 0.9  # 400 pixels

    # Ground truth with <10% IoU
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[38:58, 48:68] = 1  # Small overlap corner

    metrics = compute_pq_metrics(prob, [gt_mask], "test_low_iou")

    # Both unmatched
    assert metrics.matched_count == 0
    assert metrics.false_positive_count == 1
    assert metrics.false_negative_count == 1


def test_aggregate_pq_computes_per_image_averages():
    """Aggregate should compute correct per-image statistics."""
    # Image 1: 1 FP
    m1 = compute_pq_metrics(
        np.ones((50, 50), dtype=np.float32) * 0.9,
        [],
        "img1"
    )

    # Image 2: 1 FN
    m2 = compute_pq_metrics(
        np.zeros((50, 50), dtype=np.float32),
        [np.ones((50, 50), dtype=np.uint8)],
        "img2"
    )

    # Image 3: perfect match
    prob3 = np.zeros((50, 50), dtype=np.float32)
    prob3[10:30, 10:30] = 0.9
    gt3 = np.zeros((50, 50), dtype=np.uint8)
    gt3[10:30, 10:30] = 1
    m3 = compute_pq_metrics(prob3, [gt3], "img3")

    agg = aggregate_pq_metrics([m1, m2, m3])

    assert agg["n_images"] == 3
    assert agg["total_fp"] == 1
    assert agg["total_fn"] == 1
    assert agg["fp_per_image"] == pytest.approx(1/3)
    assert agg["fn_per_image"] == pytest.approx(1/3)
    assert agg["structural_error"] == pytest.approx(2/3)  # (1+1)/3


def test_compute_pq_validates_probability_range():
    """Out-of-range probabilities should be rejected."""
    bad_prob = np.ones((50, 50), dtype=np.float32) * 1.5

    with pytest.raises(ValueError, match="must be in"):
        compute_pq_metrics(bad_prob, [], "test")


def test_compute_pq_validates_shape_mismatch():
    """Mismatched shapes should be rejected."""
    prob = np.zeros((50, 50), dtype=np.float32)
    gt_wrong = np.zeros((60, 60), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not match"):
        compute_pq_metrics(prob, [gt_wrong], "test")


def test_compute_pq_validates_2d_input():
    """3D inputs should be rejected."""
    prob_3d = np.zeros((2, 50, 50), dtype=np.float32)

    with pytest.raises(ValueError, match="must be 2D"):
        compute_pq_metrics(prob_3d, [], "test")


def test_aggregate_pq_rejects_empty_sequence():
    """Aggregation requires at least one image."""
    with pytest.raises(ValueError, match="empty"):
        aggregate_pq_metrics([])


def test_pq_metrics_records_contract_metadata():
    """Metrics should embed contract version and scope."""
    prob = np.zeros((50, 50), dtype=np.float32)
    metrics = compute_pq_metrics(prob, [], "test")

    assert metrics.contract_version == 1
    assert metrics.metric_scope == "development_oof_instance_proxy_not_official"

    agg = aggregate_pq_metrics([metrics])
    assert agg["contract_version"] == 1
    assert agg["metric_scope"] == "development_oof_instance_proxy_not_official"
