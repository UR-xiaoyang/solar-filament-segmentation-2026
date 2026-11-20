"""Strict input validation for the M2 post-processing pipeline."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray


def _validate_probability_map(
    probability: NDArray[np.generic], *, name: str = "probability"
) -> NDArray[np.generic]:
    if not isinstance(probability, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if probability.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if probability.size == 0:
        raise ValueError(f"{name} must not be empty")
    if probability.dtype.kind not in "buif":
        raise TypeError(f"{name} dtype must be boolean or numeric")
    if not np.isfinite(probability).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(probability < 0) or np.any(probability > 1):
        raise ValueError(f"{name} values must be in the closed interval [0, 1]")
    return probability


def _validate_binary_mask(mask: NDArray[np.generic]) -> NDArray[np.bool_]:
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray")
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if mask.size == 0:
        raise ValueError("mask must not be empty")
    if mask.dtype.kind not in "buif":
        raise TypeError("mask dtype must be boolean or numeric")
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError("mask must contain only binary values 0 and 1")
    return np.asarray(mask, dtype=np.bool_)


def _validate_threshold(threshold: Real) -> float:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real number")
    value = float(threshold)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be in the closed interval [0, 1]")
    return value


def _validate_component_options(
    min_area: Integral,
    connectivity: Integral,
    max_instances: Integral | None,
) -> tuple[int, int, int | None]:
    if isinstance(min_area, (bool, np.bool_)) or not isinstance(min_area, Integral):
        raise TypeError("min_area must be an integer")
    if min_area < 1:
        raise ValueError("min_area must be at least 1")
    if isinstance(connectivity, (bool, np.bool_)) or not isinstance(
        connectivity, Integral
    ):
        raise TypeError("connectivity must be an integer")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be either 4 or 8")
    if max_instances is not None:
        if isinstance(max_instances, (bool, np.bool_)) or not isinstance(
            max_instances, Integral
        ):
            raise TypeError("max_instances must be an integer or None")
        if max_instances < 1:
            raise ValueError("max_instances must be at least 1")
        max_instances = int(max_instances)
    return int(min_area), int(connectivity), max_instances
