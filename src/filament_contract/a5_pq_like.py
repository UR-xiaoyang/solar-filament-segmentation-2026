"""A5-4 WP0-r2: Strict PQ-like evaluation for development OOF.

This module implements the strict, submission-equivalent PQ-like metric for
A5-4 architectural selection, with deterministic non-overlapping ownership
and high-IoU (>0.50) matching.

**Critical boundaries**:
- Development OOF only (never test labels or temporal holdout selection)
- Submission-equivalent instance proxy (NOT official Kaggle evaluator)
- Deterministic non-overlapping ownership before matching
- High IoU threshold (>0.50) for TP, one-to-one matching
- Annotation-record isolation (never fabricate cross-annotator instance IDs)
- Immutable artifacts with input/output hashes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


PQ_LIKE_VERSION = 1
PQ_LIKE_THRESHOLD = 0.50
PQ_LIKE_MIN_AREA = 256
PQ_LIKE_CONNECTIVITY = 8
PQ_LIKE_MAX_INSTANCES = 64
PQ_LIKE_MATCH_IOU_THRESHOLD = 0.50  # Strict: only IoU > 0.50 counts as TP

METRIC_SCOPE = "development_oof_pq_like_not_official"


@dataclass(frozen=True, slots=True)
class PQLikeMetrics:
    """Per-image strict PQ-like metrics under A5-4 WP0-r2 contract."""

    image_id: str
    prediction_count: int
    target_count: int
    tp_count: int
    fp_count: int
    fn_count: int
    sum_iou: float
    pq_like: float  # sum_iou / (tp + 0.5*fp + 0.5*fn)
    sq: float | None  # sum_iou / tp, None if tp=0
    rq: float  # tp / (tp + 0.5*fp + 0.5*fn)
    mean_matched_iou: float | None
    overlapping_pred_pixels_removed: int
    contract_version: int = PQ_LIKE_VERSION
    metric_scope: str = METRIC_SCOPE


def _deterministic_non_overlapping_ownership(
    instance_masks: Sequence[NDArray[np.bool_]],
    priority_scores: Sequence[float] | None = None,
) -> list[NDArray[np.bool_]]:
    """Assign non-overlapping ownership to instances with deterministic priority.

    Args:
        instance_masks: List of binary masks (H, W), may overlap
        priority_scores: Optional scores for priority ordering (higher = higher priority)
                        If None, uses descending area then original index

    Returns:
        List of non-overlapping binary masks in same order as input.
        Instances with zero remaining pixels after ownership are kept as empty.
    """
    if not instance_masks:
        return []

    height, width = instance_masks[0].shape
    n_instances = len(instance_masks)

    # Determine priority order
    if priority_scores is None:
        # Default: descending area, then original index for tie-break
        areas = [mask.sum() for mask in instance_masks]
        priority_order = sorted(
            range(n_instances),
            key=lambda i: (-areas[i], i),
        )
    else:
        # Descending score, then original index for tie-break
        priority_order = sorted(
            range(n_instances),
            key=lambda i: (-priority_scores[i], i),
        )

    # Allocate ownership
    owned = np.zeros((height, width), dtype=np.int32)  # 0 = unowned
    non_overlapping = []

    for idx in priority_order:
        mask = instance_masks[idx]
        # Assign pixels not yet owned
        available = (owned == 0) & mask
        owned[available] = idx + 1
        non_overlapping.append(available)

    # Return in original order
    result = [None] * n_instances
    for original_idx, non_overlap_mask in zip(priority_order, non_overlapping, strict=True):
        result[original_idx] = non_overlap_mask

    return result


def _compute_iou_matrix(
    pred_masks: Sequence[NDArray[np.bool_]],
    target_masks: Sequence[NDArray[np.bool_]],
) -> NDArray[np.floating]:
    """Compute IoU matrix between prediction and target instance masks.

    Args:
        pred_masks: Prediction binary masks (N_pred, H, W)
        target_masks: Target binary masks (N_target, H, W)

    Returns:
        IoU matrix (N_pred, N_target)
    """
    n_pred = len(pred_masks)
    n_target = len(target_masks)

    if n_pred == 0 or n_target == 0:
        return np.zeros((n_pred, n_target), dtype=np.float32)

    iou_matrix = np.zeros((n_pred, n_target), dtype=np.float32)

    for i, pred_mask in enumerate(pred_masks):
        for j, target_mask in enumerate(target_masks):
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union

    return iou_matrix


def _match_instances_hungarian(
    pred_masks: Sequence[NDArray[np.bool_]],
    target_masks: Sequence[NDArray[np.bool_]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Match instances using Hungarian algorithm with IoU threshold.

    Args:
        pred_masks: Prediction masks (non-overlapping)
        target_masks: Target masks
        iou_threshold: Minimum IoU for TP (strict: >0.50)

    Returns:
        (matches, unmatched_pred_indices, unmatched_target_indices)
        matches: List of (pred_idx, target_idx, iou)
    """
    n_pred = len(pred_masks)
    n_target = len(target_masks)

    if n_pred == 0 or n_target == 0:
        return [], set(range(n_pred)), set(range(n_target))

    # Compute IoU matrix
    iou_matrix = _compute_iou_matrix(pred_masks, target_masks)

    # Apply threshold: set below-threshold to 0
    thresholded = iou_matrix.copy()
    thresholded[thresholded <= iou_threshold] = 0.0

    # Hungarian matching on cost matrix (maximize IoU = minimize -IoU)
    cost_matrix = -thresholded
    pred_indices, target_indices = linear_sum_assignment(cost_matrix)

    # Extract valid matches
    matches = []
    matched_pred = set()
    matched_target = set()

    for pred_idx, target_idx in zip(pred_indices, target_indices, strict=True):
        iou = iou_matrix[pred_idx, target_idx]
        if iou > iou_threshold:  # Strict: only >threshold
            matches.append((pred_idx, target_idx, iou))
            matched_pred.add(pred_idx)
            matched_target.add(target_idx)

    unmatched_pred = set(range(n_pred)) - matched_pred
    unmatched_target = set(range(n_target)) - matched_target

    return matches, unmatched_pred, unmatched_target


def compute_pq_like_metrics(
    probability_map: NDArray[np.floating],
    ground_truth_masks: Sequence[NDArray[np.generic]],
    image_id: str,
    *,
    threshold: float = PQ_LIKE_THRESHOLD,
    min_area: int = PQ_LIKE_MIN_AREA,
    connectivity: int = PQ_LIKE_CONNECTIVITY,
    max_instances: int = PQ_LIKE_MAX_INSTANCES,
    match_iou_threshold: float = PQ_LIKE_MATCH_IOU_THRESHOLD,
) -> PQLikeMetrics:
    """Compute strict PQ-like metrics for one image.

    Args:
        probability_map: (H, W) predicted probability [0, 1]
        ground_truth_masks: List of (H, W) binary target masks (one per instance)
        image_id: Image identifier
        threshold: Probability threshold for instance generation
        min_area: Minimum instance area
        connectivity: Connectivity for instance generation (4 or 8)
        max_instances: Maximum instances to generate
        match_iou_threshold: IoU threshold for TP (strict: >0.50)

    Returns:
        PQLikeMetrics with strict PQ-like, SQ, RQ, TP/FP/FN
    """
    from .postprocess import probability_to_instances

    # Generate instances from probability map
    instances = probability_to_instances(
        probability_map,
        threshold=threshold,
        min_area=min_area,
        connectivity=connectivity,
        max_instances=max_instances,
    )

    # Convert to binary masks
    height, width = probability_map.shape
    pred_masks_overlapping = []
    pred_scores = []
    for instance in instances:
        # InstanceMask has .mask attribute (not dict-like)
        mask = instance.mask.astype(bool) if hasattr(instance, 'mask') else instance["mask"]
        pred_masks_overlapping.append(mask)
        pred_scores.append(instance.score if hasattr(instance, 'score') else instance["score"])

    # Deterministic non-overlapping ownership
    pred_masks_non_overlap = _deterministic_non_overlapping_ownership(
        pred_masks_overlapping,
        priority_scores=pred_scores,
    )

    # Remove empty instances after ownership
    overlapping_removed = sum(
        orig.sum() - non_overlap.sum()
        for orig, non_overlap in zip(pred_masks_overlapping, pred_masks_non_overlap, strict=True)
    )

    pred_masks = [mask for mask in pred_masks_non_overlap if mask.sum() > 0]

    # Target masks
    target_masks = [np.asarray(mask, dtype=bool) for mask in ground_truth_masks]

    # Match with strict IoU threshold
    matches, unmatched_pred, unmatched_target = _match_instances_hungarian(
        pred_masks,
        target_masks,
        match_iou_threshold,
    )

    # Compute metrics
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_target)
    sum_iou = sum(iou for _, _, iou in matches)

    denominator = tp + 0.5 * fp + 0.5 * fn
    pq_like = sum_iou / denominator if denominator > 0 else 0.0
    sq = sum_iou / tp if tp > 0 else None
    rq = tp / denominator if denominator > 0 else 0.0
    mean_matched_iou = sum_iou / tp if tp > 0 else None

    return PQLikeMetrics(
        image_id=image_id,
        prediction_count=len(pred_masks),
        target_count=len(target_masks),
        tp_count=tp,
        fp_count=fp,
        fn_count=fn,
        sum_iou=sum_iou,
        pq_like=pq_like,
        sq=sq,
        rq=rq,
        mean_matched_iou=mean_matched_iou,
        overlapping_pred_pixels_removed=overlapping_removed,
    )


def aggregate_pq_like_metrics(
    metrics_list: Sequence[PQLikeMetrics],
) -> dict[str, float | int | str]:
    """Aggregate per-image PQ-like metrics to dataset level.

    Args:
        metrics_list: List of per-image metrics

    Returns:
        Aggregated metrics dict
    """
    if not metrics_list:
        return {
            "metric_scope": METRIC_SCOPE,
            "sample_count": 0,
            "pq_like_mean": 0.0,
            "sq_mean": None,
            "rq_mean": 0.0,
        }

    n = len(metrics_list)
    total_tp = sum(m.tp_count for m in metrics_list)
    total_fp = sum(m.fp_count for m in metrics_list)
    total_fn = sum(m.fn_count for m in metrics_list)
    total_sum_iou = sum(m.sum_iou for m in metrics_list)

    # Aggregate PQ-like
    pq_like_mean = np.mean([m.pq_like for m in metrics_list])

    # Aggregate SQ (only over images with TP)
    sq_values = [m.sq for m in metrics_list if m.sq is not None]
    sq_mean = np.mean(sq_values) if sq_values else None

    # Aggregate RQ
    rq_mean = np.mean([m.rq for m in metrics_list])

    # Global PQ-like (alternative aggregation)
    global_denominator = total_tp + 0.5 * total_fp + 0.5 * total_fn
    global_pq_like = total_sum_iou / global_denominator if global_denominator > 0 else 0.0

    return {
        "metric_scope": METRIC_SCOPE,
        "contract_version": PQ_LIKE_VERSION,
        "sample_count": n,
        "pq_like_mean": float(pq_like_mean),
        "pq_like_global": float(global_pq_like),
        "sq_mean": float(sq_mean) if sq_mean is not None else None,
        "rq_mean": float(rq_mean),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "total_sum_iou": float(total_sum_iou),
        "match_iou_threshold": PQ_LIKE_MATCH_IOU_THRESHOLD,
        "threshold": PQ_LIKE_THRESHOLD,
        "min_area": PQ_LIKE_MIN_AREA,
    }
