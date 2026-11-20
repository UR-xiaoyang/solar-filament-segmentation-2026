"""Instance-level greedy matcher for the local diagnostic metrics."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .metrics_model import DiagnosticInstanceMatch, DiagnosticInstanceMetrics
from .metrics_validation import _materialise_instance_masks, _validate_iou_threshold
from .postprocess import InstanceMask


class InstanceMetricsCalculator:
    """Match predicted and true instances by deterministic greedy IoU selection."""

    @staticmethod
    def calculate(
        predictions: Sequence[InstanceMask | NDArray[np.generic]],
        targets: Sequence[InstanceMask | NDArray[np.generic]],
        *,
        match_iou_threshold: Real = 0.5,
        relation_iou_threshold: Real = 0.0,
    ) -> DiagnosticInstanceMetrics:
        match_threshold = _validate_iou_threshold(
            match_iou_threshold, name="match_iou_threshold", allow_zero=False
        )
        relation_threshold = _validate_iou_threshold(
            relation_iou_threshold, name="relation_iou_threshold", allow_zero=True
        )
        prediction_masks = _materialise_instance_masks(predictions, name="predictions")
        target_masks = _materialise_instance_masks(targets, name="targets")

        all_masks = [*prediction_masks, *target_masks]
        if all_masks:
            expected_shape = all_masks[0].shape
            if any(mask.shape != expected_shape for mask in all_masks[1:]):
                raise ValueError("all prediction and target instance shapes must match")

        prediction_areas = [int(np.count_nonzero(mask)) for mask in prediction_masks]
        target_areas = [int(np.count_nonzero(mask)) for mask in target_masks]
        pair_scores: list[tuple[float, float, int, int]] = []
        target_overlaps = [0] * len(prediction_masks)
        prediction_overlaps = [0] * len(target_masks)

        for prediction_index, prediction_mask in enumerate(prediction_masks):
            for target_index, target_mask in enumerate(target_masks):
                intersection = int(np.count_nonzero(prediction_mask & target_mask))
                area_sum = prediction_areas[prediction_index] + target_areas[target_index]
                union = area_sum - intersection
                dice = 2.0 * intersection / area_sum
                iou = intersection / union
                pair_scores.append((iou, dice, prediction_index, target_index))
                if iou > relation_threshold:
                    target_overlaps[prediction_index] += 1
                    prediction_overlaps[target_index] += 1

        matched_predictions: set[int] = set()
        matched_targets: set[int] = set()
        matches: list[DiagnosticInstanceMatch] = []
        for iou, dice, prediction_index, target_index in sorted(
            pair_scores, key=lambda item: (-item[0], -item[1], item[2], item[3])
        ):
            if iou < match_threshold:
                break
            if prediction_index in matched_predictions or target_index in matched_targets:
                continue
            matched_predictions.add(prediction_index)
            matched_targets.add(target_index)
            matches.append(
                DiagnosticInstanceMatch(
                    prediction_index=prediction_index,
                    target_index=target_index,
                    dice=dice,
                    iou=iou,
                )
            )

        matches.sort(key=lambda match: (match.prediction_index, match.target_index))
        matched_count = len(matches)
        prediction_count = len(prediction_masks)
        target_count = len(target_masks)
        return DiagnosticInstanceMetrics(
            prediction_count=prediction_count,
            target_count=target_count,
            matched_count=matched_count,
            false_positive_count=prediction_count - matched_count,
            false_negative_count=target_count - matched_count,
            signed_count_error=prediction_count - target_count,
            absolute_count_error=abs(prediction_count - target_count),
            merged_prediction_count=sum(count > 1 for count in target_overlaps),
            split_target_count=sum(count > 1 for count in prediction_overlaps),
            target_overlaps_per_prediction=tuple(target_overlaps),
            prediction_overlaps_per_target=tuple(prediction_overlaps),
            matches=tuple(matches),
            mean_matched_dice=(
                float(np.mean([match.dice for match in matches])) if matches else None
            ),
            mean_matched_iou=(
                float(np.mean([match.iou for match in matches])) if matches else None
            ),
            match_iou_threshold=match_threshold,
            relation_iou_threshold=relation_threshold,
        )


def diagnostic_instance_metrics(
    predictions: Sequence[InstanceMask | NDArray[np.generic]],
    targets: Sequence[InstanceMask | NDArray[np.generic]],
    *,
    match_iou_threshold: Real = 0.5,
    relation_iou_threshold: Real = 0.0,
) -> DiagnosticInstanceMetrics:
    """Compute transparent local instance diagnostics; this is not official.

    True target instances are required. Candidate pairs at or above
    ``match_iou_threshold`` are greedily selected by descending IoU (then Dice
    and input indices) with one prediction and one target used at most once.
    Merge/split relations count all pairs whose IoU is strictly greater than
    ``relation_iou_threshold``; the default therefore means any non-empty
    geometric overlap.
    """

    return InstanceMetricsCalculator.calculate(
        predictions,
        targets,
        match_iou_threshold=match_iou_threshold,
        relation_iou_threshold=relation_iou_threshold,
    )
