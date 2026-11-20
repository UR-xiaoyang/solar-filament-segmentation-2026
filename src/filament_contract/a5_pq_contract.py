"""A5-4 WP0: Frozen PQ measurement contract for architectural experiments.

This module defines the immutable PQ (Panoptic Quality proxy) measurement contract
used throughout the A5-4 specialization program. It provides deterministic instance
generation, matching, and structural error computation for development OOF only.

**Critical boundaries**:
- Development OOF only (never test labels or temporal holdout selection)
- Instance proxy metric (NOT official Kaggle evaluator)
- Frozen parameters (threshold, min_area, matching IoU)
- Immutable artifacts with input/output hashes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .metrics_instance import diagnostic_instance_metrics
from .postprocess import probability_to_instances, InstanceMask

# Frozen contract version and parameters
PQ_CONTRACT_VERSION = 1
PQ_THRESHOLD = 0.50
PQ_MIN_AREA = 256
PQ_CONNECTIVITY = 8
PQ_MAX_INSTANCES = 64
PQ_MATCH_IOU = 0.10
PQ_RELATION_IOU = 0.0

CONTRACT_SCOPE = "development_oof_instance_proxy_not_official"


@dataclass(frozen=True, slots=True)
class PQMetrics:
    """Per-image PQ metrics under the frozen A5-4 contract."""

    image_id: str
    prediction_count: int
    target_count: int
    matched_count: int
    false_positive_count: int
    false_negative_count: int
    split_count: int
    merge_count: int
    structural_error: float
    matched_dice_mean: float | None
    matched_iou_mean: float | None
    contract_version: int = PQ_CONTRACT_VERSION
    metric_scope: str = CONTRACT_SCOPE


def compute_pq_metrics(
    probability_map: NDArray[np.floating],
    ground_truth_masks: Sequence[NDArray[np.generic]],
    image_id: str,
) -> PQMetrics:
    """Compute PQ metrics under frozen contract for one image.

    Args:
        probability_map: (H, W) predicted probability [0, 1]
        ground_truth_masks: List of (H, W) binary ground truth instance masks
        image_id: Image identifier for record linkage

    Returns:
        PQMetrics with structural_error = FP + FN + 0.5*split + 0.5*merge

    Raises:
        ValueError: If inputs are invalid or shapes mismatch
    """
    if probability_map.ndim != 2:
        raise ValueError("probability_map must be 2D")
    if not 0 <= probability_map.min() <= probability_map.max() <= 1:
        raise ValueError("probability_map must be in [0, 1]")

    # Generate predicted instances with frozen contract parameters
    predicted_instances = probability_to_instances(
        probability_map,
        threshold=PQ_THRESHOLD,
        min_area=PQ_MIN_AREA,
        connectivity=PQ_CONNECTIVITY,
        max_instances=PQ_MAX_INSTANCES,
    )

    # Validate ground truth masks
    if ground_truth_masks:
        expected_shape = probability_map.shape
        for idx, gt_mask in enumerate(ground_truth_masks):
            if gt_mask.shape != expected_shape:
                raise ValueError(
                    f"ground truth mask {idx} shape {gt_mask.shape} "
                    f"does not match probability map shape {expected_shape}"
                )

    # Match instances using frozen IoU threshold
    instance_metrics = diagnostic_instance_metrics(
        predicted_instances,
        ground_truth_masks,
        match_iou_threshold=PQ_MATCH_IOU,
        relation_iou_threshold=PQ_RELATION_IOU,
    )

    # Compute structural error (primary A5-4 selection metric)
    structural_error = (
        instance_metrics.false_positive_count
        + instance_metrics.false_negative_count
        + 0.5 * instance_metrics.split_target_count
        + 0.5 * instance_metrics.merged_prediction_count
    )

    return PQMetrics(
        image_id=image_id,
        prediction_count=instance_metrics.prediction_count,
        target_count=instance_metrics.target_count,
        matched_count=instance_metrics.matched_count,
        false_positive_count=instance_metrics.false_positive_count,
        false_negative_count=instance_metrics.false_negative_count,
        split_count=instance_metrics.split_target_count,
        merge_count=instance_metrics.merged_prediction_count,
        structural_error=structural_error,
        matched_dice_mean=instance_metrics.mean_matched_dice,
        matched_iou_mean=instance_metrics.mean_matched_iou,
    )


def aggregate_pq_metrics(per_image_metrics: Sequence[PQMetrics]) -> dict[str, float]:
    """Aggregate per-image PQ metrics to dataset-level summary.

    Returns aggregate structural_error and per-image averages of all components.
    """
    if not per_image_metrics:
        raise ValueError("cannot aggregate empty metric sequence")

    n_images = len(per_image_metrics)
    total_fp = sum(m.false_positive_count for m in per_image_metrics)
    total_fn = sum(m.false_negative_count for m in per_image_metrics)
    total_split = sum(m.split_count for m in per_image_metrics)
    total_merge = sum(m.merge_count for m in per_image_metrics)
    total_matched = sum(m.matched_count for m in per_image_metrics)

    # Collect non-None matched Dice values
    matched_dice_values = [
        m.matched_dice_mean for m in per_image_metrics if m.matched_dice_mean is not None
    ]

    return {
        "n_images": n_images,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_split": total_split,
        "total_merge": total_merge,
        "total_matched": total_matched,
        "fp_per_image": total_fp / n_images,
        "fn_per_image": total_fn / n_images,
        "split_per_image": total_split / n_images,
        "merge_per_image": total_merge / n_images,
        "structural_error": (total_fp + total_fn + 0.5 * total_split + 0.5 * total_merge)
        / n_images,
        "matched_dice_mean": (
            float(np.mean(matched_dice_values)) if matched_dice_values else None
        ),
        "contract_version": PQ_CONTRACT_VERSION,
        "metric_scope": CONTRACT_SCOPE,
    }


__all__ = [
    "PQ_CONTRACT_VERSION",
    "PQ_THRESHOLD",
    "PQ_MIN_AREA",
    "PQ_CONNECTIVITY",
    "PQ_MAX_INSTANCES",
    "PQ_MATCH_IOU",
    "CONTRACT_SCOPE",
    "PQMetrics",
    "compute_pq_metrics",
    "aggregate_pq_metrics",
]
