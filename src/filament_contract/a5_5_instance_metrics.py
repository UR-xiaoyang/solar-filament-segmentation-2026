"""A5-5 metrics bridge: Mask R-CNN outputs → strict PQ contract.

The strict PQ contract (``a5_strict_pq_contract``) consumes per-image
probability maps at the **2048×2048 evaluation resolution** and a list
of fixed-size (H, W) binary instance masks for the ground truth.

Mask R-CNN, in contrast, returns ``Dict[image_id, {boxes, labels, scores, masks}]``
at the model-grid resolution (e.g. 768×768), where masks are already
binary and non-overlapping by construction. This module:

1. Resizes each predicted mask to the evaluation resolution.
2. Applies the deterministic ``score_threshold`` / ``mask_threshold`` /
   ``min_instance_area`` / ``max_instances`` filters.
3. Computes the strict PQ contract metrics (record-level → per-image).

The bridge is **single-responsibility**: it never touches the model or
the dataset, only the tensors returned by ``model(images)`` and the
ground-truth masks produced by the COCO index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

from .a5_pq_like import PQLikeMetrics, compute_pq_like_metrics
from .a5_strict_pq_contract import StrictPQContract


@dataclass(frozen=True)
class MaskRCNNPrediction:
    """One image's Mask R-CNN predictions, with provenance for reproducibility."""

    image_id: str
    scores: NDArray[np.float32]  # shape [N]
    boxes: NDArray[np.float32]  # shape [N, 4] (x1, y1, x2, y2) at model-grid
    masks: NDArray[np.uint8]  # shape [N, H, W] binary at model-grid
    model_grid_hw: Tuple[int, int]


def _resize_mask_to_eval(
    mask: NDArray[np.uint8],
    target_hw: Tuple[int, int],
) -> NDArray[np.uint8]:
    """Nearest-neighbor resize of a binary mask to the evaluation resolution."""
    if mask.shape == target_hw:
        return mask.astype(np.uint8, copy=False)
    zoom_h = target_hw[0] / mask.shape[0]
    zoom_w = target_hw[1] / mask.shape[1]
    resized = zoom(mask.astype(np.float32), (zoom_h, zoom_w), order=0)
    return (resized > 0.5).astype(np.uint8)


def maskrcnn_predictions_to_instance_masks(
    prediction: MaskRCNNPrediction,
    *,
    score_threshold: float,
    mask_threshold: float,
    min_instance_area: int,
    max_instances: int,
    evaluation_resolution: Tuple[int, int],
) -> List[NDArray[np.bool_]]:
    """Filter and resize Mask R-CNN predictions into instance masks.

    Applies score_threshold, mask-threshold binarization, min_instance_area
    filtering, and top-K cap. Returns a list of binary masks at the
    evaluation resolution.

    Note: torchvision Mask R-CNN masks are already sigmoid-activated
    probabilities in [0, 1], so we apply a final threshold for
    binarization. To stay consistent with the existing strict PQ
    contract (which assumes non-overlapping instances), we keep only the
    top-K by descending score and let the contract handle overlapping
    ownership via priority.
    """
    if prediction.scores.size == 0:
        return []

    eval_h, eval_w = evaluation_resolution

    keep = np.where(prediction.scores >= score_threshold)[0]
    if keep.size == 0:
        return []

    # Sort by descending score so we keep the most confident instances.
    keep = keep[np.argsort(-prediction.scores[keep])]
    if keep.size > max_instances:
        keep = keep[:max_instances]

    out: List[NDArray[np.bool_]] = []
    for idx in keep:
        binary_mask_at_grid = (prediction.masks[idx] >= mask_threshold)
        if binary_mask_at_grid.sum() < min_instance_area:
            continue
        resized = _resize_mask_to_eval(
            binary_mask_at_grid.astype(np.uint8), evaluation_resolution
        )
        if resized.sum() < min_instance_area:
            continue
        out.append(resized.astype(bool))
    return out


def evaluate_maskrcnn_predictions(
    *,
    predictions: Sequence[MaskRCNNPrediction],
    ground_truth_by_image: Dict[str, List[NDArray[np.bool_]]],
    contract: StrictPQContract,
) -> List[Tuple[PQLikeMetrics, str]]:
    """Compute per-image strict PQ-like metrics for Mask R-CNN outputs.

    Args:
        predictions: One MaskRCNNPrediction per image.
        ground_truth_by_image: ``image_id -> list of (H, W) binary masks``
            at the evaluation resolution.
        contract: The strict PQ contract.

    Returns:
        List of (PQLikeMetrics, image_id). One entry per image, in the
        same order as ``predictions``.
    """
    eval_h, eval_w = contract.evaluation_resolution
    out: List[Tuple[PQLikeMetrics, str]] = []
    for prediction in predictions:
        instance_masks = maskrcnn_predictions_to_instance_masks(
            prediction,
            score_threshold=contract.probability_threshold,
            mask_threshold=contract.probability_threshold,  # mask binarization at 0.5
            min_instance_area=contract.min_instance_area,
            max_instances=contract.max_instances,
            evaluation_resolution=contract.evaluation_resolution,
        )

        # The strict PQ contract expects a probability map. For Mask
        # R-CNN we synthesize one by stacking instance masks and
        # clipping to [0, 1]. This is exactly equivalent to the
        # non-overlapping union of the kept instances — which is what
        # ``probability_to_instances`` would re-discover via
        # threshold + connected components, so the resulting PQ is
        # directly comparable to the Control-768 baseline.
        if instance_masks:
            stacked = np.stack(instance_masks, axis=0).astype(np.float32)
            probability = np.clip(stacked.sum(axis=0), 0.0, 1.0).astype(np.float32)
        else:
            probability = np.zeros(contract.evaluation_resolution, dtype=np.float32)

        gt_masks = ground_truth_by_image.get(prediction.image_id, [])
        metrics = compute_pq_like_metrics(
            probability_map=probability,
            ground_truth_masks=gt_masks,
            image_id=prediction.image_id,
            threshold=contract.probability_threshold,
            min_area=contract.min_instance_area,
            connectivity=contract.connectivity,
            max_instances=contract.max_instances,
            match_iou_threshold=contract.match_iou_threshold,
        )
        out.append((metrics, prediction.image_id))
    return out


__all__ = [
    "MaskRCNNPrediction",
    "evaluate_maskrcnn_predictions",
    "maskrcnn_predictions_to_instance_masks",
]