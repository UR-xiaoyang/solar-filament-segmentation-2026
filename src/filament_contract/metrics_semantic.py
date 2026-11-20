"""Semantic overlap calculator for the local diagnostic metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .metrics_model import DiagnosticSemanticMetrics
from .metrics_validation import _as_binary_mask


class SemanticMetricsCalculator:
    """Compute per-image semantic Dice and IoU; this is not the official score."""

    @staticmethod
    def calculate(
        prediction: NDArray[np.generic], target: NDArray[np.generic]
    ) -> DiagnosticSemanticMetrics:
        prediction_mask = _as_binary_mask(prediction, name="prediction")
        target_mask = _as_binary_mask(target, name="target")
        if prediction_mask.shape != target_mask.shape:
            raise ValueError("prediction and target shapes must match")

        prediction_area = int(np.count_nonzero(prediction_mask))
        target_area = int(np.count_nonzero(target_mask))
        intersection = int(np.count_nonzero(prediction_mask & target_mask))
        area_sum = prediction_area + target_area
        union = prediction_area + target_area - intersection
        dice = 1.0 if area_sum == 0 else 2.0 * intersection / area_sum
        iou = 1.0 if union == 0 else intersection / union
        return DiagnosticSemanticMetrics(
            prediction_area=prediction_area,
            target_area=target_area,
            intersection=intersection,
            dice=dice,
            iou=iou,
        )


def diagnostic_semantic_metrics(
    prediction: NDArray[np.generic], target: NDArray[np.generic]
) -> DiagnosticSemanticMetrics:
    """Compute local semantic Dice and IoU; this is not the official score.

    The local empty-mask policy is explicit: two empty masks score 1.0, while
    exactly one empty mask scores 0.0.
    """

    return SemanticMetricsCalculator.calculate(prediction, target)
