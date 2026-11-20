"""Union-find connected-component extraction from a binary mask.

Components are computed without SciPy using row runs and a union-find
structure, then filtered and ordered deterministically.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .postprocess_model import InstanceMask, _Run
from .postprocess_validation import (
    _validate_binary_mask,
    _validate_component_options,
    _validate_probability_map,
)


def _find(parent: list[int], label: int) -> int:
    root = label
    while parent[root] != root:
        root = parent[root]
    while parent[label] != label:
        next_label = parent[label]
        parent[label] = root
        label = next_label
    return root


def _union(parent: list[int], first: int, second: int) -> None:
    first_root = _find(parent, first)
    second_root = _find(parent, second)
    if first_root == second_root:
        return
    smaller, larger = sorted((first_root, second_root))
    parent[larger] = smaller


def _row_runs(row: NDArray[np.bool_], row_index: int, first_label: int) -> list[_Run]:
    padded = np.pad(row.astype(np.int8, copy=False), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [
        _Run(row_index, int(start), int(end), first_label + offset)
        for offset, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]


def _runs_touch(previous: _Run, current: _Run, connectivity: int) -> bool:
    if connectivity == 4:
        return previous.start < current.end and current.start < previous.end
    return previous.start <= current.end and current.start <= previous.end


def _previous_run_is_before(previous: _Run, current: _Run, connectivity: int) -> bool:
    if connectivity == 4:
        return previous.end <= current.start
    return previous.end < current.start


def _current_run_is_before(previous: _Run, current: _Run, connectivity: int) -> bool:
    if connectivity == 4:
        return previous.start >= current.end
    return previous.start > current.end


class ComponentExtractor:
    """Extract filtered, deterministically ordered instances from a binary mask."""

    @staticmethod
    def extract(
        mask: NDArray[np.generic],
        *,
        min_area: Integral = 1,
        connectivity: Integral = 8,
        max_instances: Integral | None = None,
        score_map: NDArray[np.generic] | None = None,
    ) -> list[InstanceMask]:
        foreground = _validate_binary_mask(mask)
        minimum_area, neighbourhood, maximum_instances = _validate_component_options(
            min_area, connectivity, max_instances
        )
        if score_map is None:
            scores = None
        else:
            scores = _validate_probability_map(score_map, name="score_map")
            if scores.shape != foreground.shape:
                raise ValueError("score_map shape must match mask shape")

        all_runs: list[_Run] = []
        parent: list[int] = []
        previous_runs: list[_Run] = []

        for row_index, row in enumerate(foreground):
            current_runs = _row_runs(row, row_index, len(parent))
            parent.extend(range(len(parent), len(parent) + len(current_runs)))

            previous_start = 0
            for current in current_runs:
                while previous_start < len(previous_runs) and _previous_run_is_before(
                    previous_runs[previous_start], current, neighbourhood
                ):
                    previous_start += 1

                candidate = previous_start
                while candidate < len(previous_runs):
                    previous = previous_runs[candidate]
                    if _current_run_is_before(previous, current, neighbourhood):
                        break
                    if _runs_touch(previous, current, neighbourhood):
                        _union(parent, previous.label, current.label)
                    candidate += 1

            all_runs.extend(current_runs)
            previous_runs = current_runs

        runs_by_root: dict[int, list[_Run]] = {}
        for run in all_runs:
            root = _find(parent, run.label)
            runs_by_root.setdefault(root, []).append(run)

        components: list[tuple[int, int, float, list[_Run]]] = []
        width = foreground.shape[1]
        for runs in runs_by_root.values():
            area = sum(run.end - run.start for run in runs)
            if area < minimum_area:
                continue
            first_position = min(run.row * width + run.start for run in runs)
            if scores is None:
                score = 1.0
            else:
                score_sum = sum(
                    float(scores[run.row, run.start : run.end].sum(dtype=np.float64))
                    for run in runs
                )
                score = score_sum / area
            components.append((area, first_position, score, runs))

        components.sort(key=lambda component: (-component[0], component[1]))
        if maximum_instances is not None:
            components = components[:maximum_instances]
        instances: list[InstanceMask] = []
        for instance_id, (area, _first_position, score, runs) in enumerate(
            components, start=1
        ):
            component_mask = np.zeros(foreground.shape, dtype=np.uint8)
            for run in runs:
                component_mask[run.row, run.start : run.end] = 1
            instances.append(
                InstanceMask(
                    instance_id=instance_id,
                    mask=component_mask,
                    area=area,
                    score=score,
                )
            )
        return instances


def binary_mask_to_instances(
    mask: NDArray[np.generic],
    *,
    min_area: Integral = 1,
    connectivity: Integral = 8,
    max_instances: Integral | None = None,
    score_map: NDArray[np.generic] | None = None,
) -> list[InstanceMask]:
    """Convert a binary mask into filtered, deterministically ordered instances.

    Connected components are computed without SciPy using row runs and a union-find
    structure. With no ``score_map``, every retained component receives score 1.0.
    """

    return ComponentExtractor.extract(
        mask,
        min_area=min_area,
        connectivity=connectivity,
        max_instances=max_instances,
        score_map=score_map,
    )
