"""
A5 Strict PQ-like Contract v1.0

Canonical instance evaluation protocol for A5-4 solar filament specialization.

Key principles:
- Evaluation resolution: 2048×2048 (predictions resized from model grid)
- Ground truth unit: annotation record (not merged across annotators)
- Matching: Hungarian one-to-one, IoU > 0.50
- Prediction ownership: deterministic non-overlap before matching
- Aggregation: record-level → physical-image macro → global

This supersedes legacy a5_pq_contract.py (IoU ≥ 0.10 structural diagnostics).
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

from filament_contract.a5_pq_like import (
    PQLikeMetrics,
    compute_pq_like_metrics,
    aggregate_pq_like_metrics,
)
from filament_contract.a5_structural_diagnostics import (
    StructuralDiagnostics,
    compute_structural_diagnostics,
)


@dataclass(frozen=True)
class StrictPQContract:
    """Strict PQ-like evaluation contract parameters."""

    version: str = "1.0"
    evaluation_resolution: Tuple[int, int] = (2048, 2048)
    probability_threshold: float = 0.5
    min_instance_area: int = 256
    connectivity: int = 8
    max_instances: int = 64
    match_iou_threshold: float = 0.5  # Strict: > 0.50 for TP

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "evaluation_resolution": list(self.evaluation_resolution),
            "probability_threshold": self.probability_threshold,
            "min_instance_area": self.min_instance_area,
            "connectivity": self.connectivity,
            "max_instances": self.max_instances,
            "match_iou_threshold": self.match_iou_threshold,
        }


def load_and_validate_oof_probability(
    npz_path: Path,
    expected_image_id: str,
) -> Tuple[NDArray[np.float32], Tuple[int, int]]:
    """
    Load OOF probability from .npz and validate against M2 contract.

    Args:
        npz_path: Path to <image_id>.npz file
        expected_image_id: Expected image_id for validation

    Returns:
        (probability_map, model_shape)

    Raises:
        ValueError: If validation fails
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"OOF probability not found: {npz_path}")

    with np.load(npz_path) as data:
        # Validate required keys
        if 'probability' not in data:
            raise ValueError(f"Missing 'probability' in {npz_path}")
        if 'image_id' not in data:
            raise ValueError(f"Missing 'image_id' in {npz_path}")
        if 'model_shape' not in data:
            raise ValueError(f"Missing 'model_shape' in {npz_path}")

        probability = data['probability']
        image_id = str(data['image_id'])
        model_shape = tuple(data['model_shape'])

        # Validate image_id
        if image_id != expected_image_id:
            raise ValueError(
                f"Image ID mismatch: expected {expected_image_id}, got {image_id}"
            )

        # Validate probability format
        if probability.ndim != 2:
            raise ValueError(
                f"Probability must be 2-D, got shape {probability.shape}"
            )
        if probability.dtype != np.float32:
            raise ValueError(
                f"Probability must be float32, got {probability.dtype}"
            )
        if not np.all(np.isfinite(probability)):
            raise ValueError("Probability contains non-finite values")
        if not (np.all(probability >= 0) and np.all(probability <= 1)):
            raise ValueError("Probability must be in [0, 1]")

        # Validate model_shape matches probability
        if probability.shape != model_shape:
            raise ValueError(
                f"model_shape {model_shape} != probability.shape {probability.shape}"
            )

        return probability.astype(np.float32), model_shape


def resize_probability_to_evaluation_resolution(
    probability: NDArray[np.float32],
    target_resolution: Tuple[int, int] = (2048, 2048),
) -> NDArray[np.float32]:
    """
    Resize probability map to evaluation resolution using bilinear interpolation.

    This is the canonical M2 resize: model grid → 2048×2048.

    Args:
        probability: Model-grid probability map (H, W)
        target_resolution: Target (height, width), default 2048×2048

    Returns:
        Resized probability map at evaluation resolution
    """
    if probability.shape == target_resolution:
        return probability

    zoom_h = target_resolution[0] / probability.shape[0]
    zoom_w = target_resolution[1] / probability.shape[1]

    resized = zoom(probability, (zoom_h, zoom_w), order=1)  # bilinear

    # Clip to [0, 1] after interpolation
    resized = np.clip(resized, 0.0, 1.0)

    return resized.astype(np.float32)


def rasterize_annotation_record_masks(
    annotation_masks: Sequence[NDArray[np.bool_]],
    target_resolution: Tuple[int, int] = (2048, 2048),
) -> List[NDArray[np.bool_]]:
    """
    Rasterize annotation record masks to evaluation resolution.

    Args:
        annotation_masks: List of ground truth instance masks (any resolution)
        target_resolution: Target (height, width), default 2048×2048

    Returns:
        List of masks resized to evaluation resolution
    """
    resized_masks = []

    for mask in annotation_masks:
        if mask.shape == target_resolution:
            resized_masks.append(mask)
        else:
            zoom_h = target_resolution[0] / mask.shape[0]
            zoom_w = target_resolution[1] / mask.shape[1]
            resized = zoom(mask.astype(np.float32), (zoom_h, zoom_w), order=1) > 0.5
            resized_masks.append(resized)

    return resized_masks


def compute_input_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_strict_pq_for_record(
    probability_2048: NDArray[np.float32],
    gt_masks_2048: List[NDArray[np.bool_]],
    image_id: str,
    annotation_record_id: str,
    contract: StrictPQContract,
) -> Tuple[PQLikeMetrics, StructuralDiagnostics]:
    """
    Compute strict PQ-like and structural diagnostics for one annotation record.

    Args:
        probability_2048: Prediction probability at 2048×2048
        gt_masks_2048: Ground truth masks for this record at 2048×2048
        image_id: Physical image identifier
        annotation_record_id: Annotation record identifier
        contract: Evaluation contract parameters

    Returns:
        (pq_like_metrics, structural_diagnostics)
    """
    record_image_id = f"{image_id}__record_{annotation_record_id}"

    # Compute PQ-like
    pq_metrics = compute_pq_like_metrics(
        probability_map=probability_2048,
        ground_truth_masks=gt_masks_2048,
        image_id=record_image_id,
        threshold=contract.probability_threshold,
        min_area=contract.min_instance_area,
        match_iou_threshold=contract.match_iou_threshold,
    )

    # Compute structural diagnostics
    struct_metrics = compute_structural_diagnostics(
        probability_map=probability_2048,
        ground_truth_masks=gt_masks_2048,
        image_id=record_image_id,
        threshold=contract.probability_threshold,
        min_area=contract.min_instance_area,
    )

    return pq_metrics, struct_metrics


def aggregate_record_to_physical_image(
    record_pq_metrics: List[PQLikeMetrics],
) -> PQLikeMetrics:
    """
    Aggregate record-level PQ-like metrics to physical-image level.

    Uses macro averaging: mean of per-record metrics.

    Args:
        record_pq_metrics: List of PQ-like metrics for each annotation record

    Returns:
        Aggregated PQ-like metrics for the physical image
    """
    if not record_pq_metrics:
        raise ValueError("No record metrics to aggregate")

    # Use first record's image_id (strip record suffix)
    base_image_id = record_pq_metrics[0].image_id.split("__record_")[0]

    # Macro average
    pq_like_mean = float(np.mean([m.pq_like for m in record_pq_metrics]))
    sq_values = [m.sq for m in record_pq_metrics if m.sq is not None]
    sq_mean = float(np.mean(sq_values)) if sq_values else None
    rq_mean = float(np.mean([m.rq for m in record_pq_metrics]))

    # Sum counts
    total_pred = sum(m.prediction_count for m in record_pq_metrics)
    total_target = sum(m.target_count for m in record_pq_metrics)
    total_tp = sum(m.tp_count for m in record_pq_metrics)
    total_fp = sum(m.fp_count for m in record_pq_metrics)
    total_fn = sum(m.fn_count for m in record_pq_metrics)
    total_sum_iou = sum(m.sum_iou for m in record_pq_metrics)

    return PQLikeMetrics(
        image_id=base_image_id,
        prediction_count=total_pred,
        target_count=total_target,
        tp_count=total_tp,
        fp_count=total_fp,
        fn_count=total_fn,
        sum_iou=total_sum_iou,
        pq_like=pq_like_mean,
        sq=sq_mean,
        rq=rq_mean,
        mean_matched_iou=None,  # Not meaningful after aggregation
        overlapping_pred_pixels_removed=sum(
            m.overlapping_pred_pixels_removed for m in record_pq_metrics
        ),
        contract_version=record_pq_metrics[0].contract_version,
        metric_scope="physical_image_aggregate_from_records",
    )
