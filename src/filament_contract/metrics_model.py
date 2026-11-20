"""Data model for local diagnostic metrics that are not the official evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field

DIAGNOSTIC_METRIC_SCOPE = "local_diagnostic_not_official"


@dataclass(frozen=True, slots=True)
class DiagnosticSemanticMetrics:
    """Per-image semantic overlap under the documented local empty-mask policy."""

    prediction_area: int
    target_area: int
    intersection: int
    dice: float
    iou: float
    metric_scope: str = field(default=DIAGNOSTIC_METRIC_SCOPE, init=False)


@dataclass(frozen=True, slots=True)
class DiagnosticInstanceMatch:
    """One pair selected by the deterministic local greedy matcher."""

    prediction_index: int
    target_index: int
    dice: float
    iou: float


@dataclass(frozen=True, slots=True)
class DiagnosticInstanceMetrics:
    """Instance counts, matches, and merge/split relations for one image."""

    prediction_count: int
    target_count: int
    matched_count: int
    false_positive_count: int
    false_negative_count: int
    signed_count_error: int
    absolute_count_error: int
    merged_prediction_count: int
    split_target_count: int
    target_overlaps_per_prediction: tuple[int, ...]
    prediction_overlaps_per_target: tuple[int, ...]
    matches: tuple[DiagnosticInstanceMatch, ...]
    mean_matched_dice: float | None
    mean_matched_iou: float | None
    match_iou_threshold: float
    relation_iou_threshold: float
    matching_policy: str = field(default="greedy_descending_iou", init=False)
    metric_scope: str = field(default=DIAGNOSTIC_METRIC_SCOPE, init=False)
