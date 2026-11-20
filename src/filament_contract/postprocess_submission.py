"""Build canonical submission records from post-processed instances."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from .postprocess_model import InstanceMask
from .postprocess_instances import probability_to_instances
from .submission import SubmissionRecord, record_from_mask


def _instance_sort_key(instance: InstanceMask) -> tuple[int, int]:
    if not isinstance(instance, InstanceMask):
        raise TypeError("each instance must be an InstanceMask")
    foreground_positions = np.flatnonzero(instance.mask)
    if foreground_positions.size == 0:
        raise ValueError("instance masks must contain at least one foreground pixel")
    if int(foreground_positions.size) != instance.area:
        raise ValueError("instance area does not match its mask")
    return -instance.area, int(foreground_positions[0])


class SubmissionRecordBuilder:
    """Build submission rows with deterministic ``<image_id>_<1-based>`` IDs."""

    @staticmethod
    def build(image_id: str, instances: Sequence[InstanceMask]) -> list[SubmissionRecord]:
        if isinstance(instances, (str, bytes)) or not isinstance(instances, Sequence):
            raise TypeError("instances must be a sequence of InstanceMask objects")
        ordered = sorted(instances, key=_instance_sort_key)
        return [
            record_from_mask(image_id, f"{image_id}_{index}", instance.mask)
            for index, instance in enumerate(ordered, start=1)
        ]


def instances_to_submission_records(
    image_id: str, instances: Sequence[InstanceMask]
) -> list[SubmissionRecord]:
    """Build submission rows with deterministic ``<image_id>_<1-based>`` IDs."""

    return SubmissionRecordBuilder.build(image_id, instances)


def probability_to_submission_records(
    image_id: str,
    probability: NDArray[np.generic],
    *,
    threshold: Real = 0.5,
    min_area: Integral = 1,
    connectivity: Integral = 8,
    max_instances: Integral | None = None,
) -> list[SubmissionRecord]:
    """Run the complete probability-to-submission-row M2 post-processing path."""

    instances = probability_to_instances(
        probability,
        threshold=threshold,
        min_area=min_area,
        connectivity=connectivity,
        max_instances=max_instances,
    )
    return instances_to_submission_records(image_id, instances)
