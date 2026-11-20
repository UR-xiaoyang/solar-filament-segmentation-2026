"""Probability-map thresholding for the M2 post-processing pipeline."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .postprocess_validation import _validate_probability_map, _validate_threshold


class ProbabilityThresholder:
    """Threshold a finite [0, 1] probability map, including exact ties."""

    @staticmethod
    def apply(
        probability: NDArray[np.generic], *, threshold: Real = 0.5
    ) -> NDArray[np.uint8]:
        checked = _validate_probability_map(probability)
        cutoff = _validate_threshold(threshold)
        return np.ascontiguousarray(checked >= cutoff, dtype=np.uint8)


def threshold_probability_map(
    probability: NDArray[np.generic], *, threshold: Real = 0.5
) -> NDArray[np.uint8]:
    """Threshold a finite [0, 1] probability map, including exact ties."""

    return ProbabilityThresholder.apply(probability, threshold=threshold)
