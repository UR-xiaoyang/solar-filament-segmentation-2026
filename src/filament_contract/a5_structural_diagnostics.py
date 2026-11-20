"""A5-4 WP0-r2: Structural error diagnostics (legacy metric).

This module provides the historical structural error metric from A5-1
as an auxiliary diagnostic for A5-4. It uses low IoU threshold (≥0.10)
and should NOT be used as the primary selection metric.

**Purpose**: Explain FP/FN/split/merge changes and provide historical reference.
**Not for**: Architectural Go/No-Go decisions (use strict PQ-like instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


STRUCTURAL_DIAGNOSTICS_VERSION = 1
STRUCTURAL_MATCH_IOU = 0.10  # Historical low threshold
STRUCTURAL_RELATION_IOU = 0.0

METRIC_SCOPE = "auxiliary_structural_diagnostics_not_primary"


@dataclass(frozen=True, slots=True)
class StructuralDiagnostics:
    """Per-image structural error diagnostics (auxiliary, not primary metric)."""

    image_id: str
    prediction_count: int
    target_count: int
    matched_count: int
    false_positive_count: int
    false_negative_count: int
    split_count: int
    merge_count: int
    structural_error: float  # FP + FN + 0.5*split + 0.5*merge
    matched_dice_mean: float | None
    matched_iou_mean: float | None
    diagnostics_version: int = STRUCTURAL_DIAGNOSTICS_VERSION
    metric_scope: str = METRIC_SCOPE


def compute_structural_diagnostics(
    probability_map: NDArray[np.floating],
    ground_truth_masks: Sequence[NDArray[np.generic]],
    image_id: str,
    *,
    threshold: float = 0.50,
    min_area: int = 256,
    connectivity: int = 8,
    max_instances: int = 64,
) -> StructuralDiagnostics:
    """Compute structural error diagnostics (auxiliary metric).

    Args:
        probability_map: (H, W) predicted probability [0, 1]
        ground_truth_masks: List of (H, W) binary target masks
        image_id: Image identifier
        threshold: Probability threshold for instance generation
        min_area: Minimum instance area
        connectivity: Connectivity for instance generation
        max_instances: Maximum instances to generate

    Returns:
        StructuralDiagnostics with FP/FN/split/merge counts
    """
    from .postprocess import probability_to_instances
    from .metrics_instance import diagnostic_instance_metrics

    # Generate instances
    instances = probability_to_instances(
        probability_map,
        threshold=threshold,
        min_area=min_area,
        connectivity=connectivity,
        max_instances=max_instances,
    )

    # Convert to binary masks
    height, width = probability_map.shape
    pred_masks = []
    for instance in instances:
        # InstanceMask has .mask attribute (not dict-like)
        mask = instance.mask.astype(bool) if hasattr(instance, 'mask') else instance["mask"]
        pred_masks.append(mask)

    # Target masks
    target_masks = [np.asarray(mask, dtype=bool) for mask in ground_truth_masks]

    # Compute with low IoU threshold (historical)
    metrics = diagnostic_instance_metrics(
        pred_masks,
        target_masks,
        match_iou_threshold=STRUCTURAL_MATCH_IOU,
        relation_iou_threshold=STRUCTURAL_RELATION_IOU,
    )

    # Structural error formula
    structural_error = (
        metrics.false_positive_count
        + metrics.false_negative_count
        + 0.5 * metrics.split_target_count
        + 0.5 * metrics.merged_prediction_count
    )

    return StructuralDiagnostics(
        image_id=image_id,
        prediction_count=metrics.prediction_count,
        target_count=metrics.target_count,
        matched_count=metrics.matched_count,
        false_positive_count=metrics.false_positive_count,
        false_negative_count=metrics.false_negative_count,
        split_count=metrics.split_target_count,
        merge_count=metrics.merged_prediction_count,
        structural_error=structural_error,
        matched_dice_mean=metrics.mean_matched_dice,
        matched_iou_mean=metrics.mean_matched_iou,
    )
