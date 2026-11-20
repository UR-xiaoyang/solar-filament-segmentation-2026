"""Data model for deterministic semantic-to-instance post-processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class InstanceMask:
    """One connected foreground component produced by post-processing.

    ``instance_id`` is one-based and reflects the deterministic component order:
    descending area, followed by the first foreground pixel in row-major order.
    ``score`` is the mean source probability over the component.
    """

    instance_id: int
    mask: NDArray[np.uint8]
    area: int
    score: float


@dataclass(frozen=True, slots=True)
class _Run:
    row: int
    start: int
    end: int
    label: int
