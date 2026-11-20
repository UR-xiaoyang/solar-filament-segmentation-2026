"""Probability-map to connected-instance pipeline for M2 post-processing."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from .postprocess_components import ComponentExtractor, binary_mask_to_instances
from .postprocess_model import InstanceMask
from .postprocess_threshold import ProbabilityThresholder, threshold_probability_map
from .postprocess_validation import _validate_probability_map


class ProbabilityInstancer:
    """Threshold a probability map and convert it to connected instances."""

    @staticmethod
    def extract(
        probability: NDArray[np.generic],
        *,
        threshold: Real = 0.5,
        min_area: Integral = 1,
        connectivity: Integral = 8,
        max_instances: Integral | None = None,
    ) -> list[InstanceMask]:
        checked = _validate_probability_map(probability)
        binary = ProbabilityThresholder.apply(checked, threshold=threshold)
        return ComponentExtractor.extract(
            binary,
            min_area=min_area,
            connectivity=connectivity,
            max_instances=max_instances,
            score_map=checked,
        )


def probability_to_instances(
    probability: NDArray[np.generic],
    *,
    threshold: Real = 0.5,
    min_area: Integral = 1,
    connectivity: Integral = 8,
    max_instances: Integral | None = None,
) -> list[InstanceMask]:
    """Threshold a probability map and convert it to connected instances."""

    return ProbabilityInstancer.extract(
        probability,
        threshold=threshold,
        min_area=min_area,
        connectivity=connectivity,
        max_instances=max_instances,
    )
