"""Input validation helpers shared by the diagnostic metric calculators."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .postprocess import InstanceMask


def _as_binary_mask(
    value: InstanceMask | NDArray[np.generic], *, name: str
) -> NDArray[np.bool_]:
    mask = value.mask if isinstance(value, InstanceMask) else value
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray or InstanceMask")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if mask.size == 0:
        raise ValueError(f"{name} must not be empty")
    if mask.dtype.kind not in "buif":
        raise TypeError(f"{name} dtype must be boolean or numeric")
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError(f"{name} must contain only binary values 0 and 1")
    return np.asarray(mask, dtype=np.bool_)


def _validate_iou_threshold(value: Real, *, name: str, allow_zero: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    threshold = float(value)
    lower_bound_valid = threshold >= 0 if allow_zero else threshold > 0
    if not np.isfinite(threshold) or not lower_bound_valid or threshold > 1:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return threshold


def _materialise_instance_masks(
    values: Sequence[InstanceMask | NDArray[np.generic]], *, name: str
) -> list[NDArray[np.bool_]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of masks or InstanceMask objects")
    masks = [
        _as_binary_mask(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    ]
    for index, mask in enumerate(masks):
        if not mask.any():
            raise ValueError(f"{name}[{index}] must contain foreground pixels")
    return masks
