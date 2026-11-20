"""Deterministic semantic-to-instance post-processing for the M2 baseline.

Facade re-exporting the model, validation, extraction, and submission
submodules that implement the pipeline.
"""

from __future__ import annotations

from .postprocess_components import (
    ComponentExtractor,
    _current_run_is_before,
    _find,
    _previous_run_is_before,
    _row_runs,
    _runs_touch,
    _union,
    binary_mask_to_instances,
)
from .postprocess_instances import ProbabilityInstancer, probability_to_instances
from .postprocess_model import InstanceMask, _Run
from .postprocess_submission import (
    SubmissionRecordBuilder,
    _instance_sort_key,
    instances_to_submission_records,
    probability_to_submission_records,
)
from .postprocess_threshold import ProbabilityThresholder, threshold_probability_map
from .postprocess_validation import (
    _validate_binary_mask,
    _validate_component_options,
    _validate_probability_map,
    _validate_threshold,
)

__all__ = [
    "InstanceMask",
    "threshold_probability_map",
    "binary_mask_to_instances",
    "probability_to_instances",
    "instances_to_submission_records",
    "probability_to_submission_records",
]
