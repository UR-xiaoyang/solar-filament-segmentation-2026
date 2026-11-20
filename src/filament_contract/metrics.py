"""Local diagnostic metrics that are explicitly not the official evaluator.

Facade re-exporting the calculators and data model from the metrics submodules.
"""

from __future__ import annotations

from .metrics_instance import InstanceMetricsCalculator, diagnostic_instance_metrics
from .metrics_model import (
    DIAGNOSTIC_METRIC_SCOPE,
    DiagnosticInstanceMatch,
    DiagnosticInstanceMetrics,
    DiagnosticSemanticMetrics,
)
from .metrics_semantic import SemanticMetricsCalculator, diagnostic_semantic_metrics
from .metrics_validation import (
    _as_binary_mask,
    _materialise_instance_masks,
    _validate_iou_threshold,
)

__all__ = [
    "DIAGNOSTIC_METRIC_SCOPE",
    "DiagnosticSemanticMetrics",
    "DiagnosticInstanceMatch",
    "DiagnosticInstanceMetrics",
    "diagnostic_semantic_metrics",
    "diagnostic_instance_metrics",
]
