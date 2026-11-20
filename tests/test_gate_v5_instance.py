"""Contract tests for the FS-3 instance gate (v5) and its matcher.

Gate v5 decides whether a post-processing route continues, so its classification
branches and the greedy matcher that feeds it must be pinned by tests rather
than by remembered numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from gate_v5_instance import MATERIAL_REGRESSION_PQ, classify  # noqa: E402
from instance_score_oof import _fuse_thresholds, _match  # noqa: E402
from run_instance_routes import BASELINE_KNOBS, ROUTES  # noqa: E402

SCORER_KNOBS = {"threshold", "min_area", "max_area", "smooth_sigma",
                "open_radius", "score_floor", "top_k", "split_min_area",
                "seed_threshold", "reject_border", "min_elongation",
                "quality_model", "quality_threshold", "dual_threshold", "link_iou"}


def record(**overrides):
    base = {
        "route_id": "IN99",
        "stage": "screen",
        "metric": {"name": "pq", "primary_delta": 0.001},
        "secondary": {"fp_relative_delta": 0.01, "dice_delta": -0.001},
        "coverage_complete": True,
        "contract_valid": True,
    }
    base.update(overrides)
    return base


def test_positive_pq_is_a_screen_signal() -> None:
    assert classify(record()).status == "research_signal"


def test_regression_beyond_band_stops_the_configuration() -> None:
    decision = classify(record(metric={"name": "pq",
                                       "primary_delta": -(MATERIAL_REGRESSION_PQ + 0.001)}))
    assert decision.status == "clear_failure"


def test_delta_inside_the_band_stays_open() -> None:
    decision = classify(record(metric={"name": "pq", "primary_delta": -0.001}))
    assert decision.status == "inconclusive"


def test_incomplete_coverage_is_not_scientific_evidence() -> None:
    assert classify(record(coverage_complete=False)).status == "technical_invalid"


def test_an_inert_knob_is_not_evidence_against_the_mechanism() -> None:
    decision = classify(record(metric={"name": "pq", "primary_delta": 0.0}, triggered=False))
    assert decision.status == "not_triggered"


def test_confirmation_sign_reversal_is_not_reproduced() -> None:
    decision = classify(record(stage="confirmation",
                               metric={"name": "pq", "primary_delta": -0.0005},
                               prior_primary_delta=0.002))
    assert decision.status == "nonreproduced_inconclusive"


def test_pq_up_with_material_dice_drop_is_a_metric_conflict() -> None:
    decision = classify(record(metric={"name": "pq", "primary_delta": 0.002},
                               secondary={"fp_relative_delta": 0.0, "dice_delta": -0.02}))
    assert decision.status == "metric_conflict"


def test_fp_explosion_without_pq_gain_stops() -> None:
    decision = classify(record(metric={"name": "pq", "primary_delta": 0.0},
                               secondary={"fp_relative_delta": 0.25, "dice_delta": 0.0}))
    assert decision.status == "clear_failure"


def _mask(rows, cols, shape=(8, 8)):
    mask = np.zeros(shape, dtype=bool)
    mask[np.ix_(rows, cols)] = True
    return mask


def test_matcher_finds_the_one_to_one_pair() -> None:
    pred = [_mask([0, 1], [0, 1])]
    gt = [_mask([0, 1], [0, 1])]
    result = _match(pred, gt, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
    assert result["iou_sum"] == 1.0


def test_matcher_counts_unmatched_predictions_and_targets() -> None:
    pred = [_mask([0, 1], [0, 1]), _mask([5], [5])]
    gt = [_mask([0, 1], [0, 1]), _mask([7], [7])]
    result = _match(pred, gt, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)


def test_matcher_respects_the_iou_threshold() -> None:
    pred = [_mask([0, 1], [0, 1])]
    gt = [_mask([0, 1], [0])]
    assert _match(pred, gt, 0.9)["tp"] == 0
    assert _match(pred, gt, 0.5)["tp"] == 1


def test_instance_routes_declare_one_known_knob_each() -> None:
    ids = [route["id"] for route in ROUTES]
    assert len(ids) == len(set(ids))
    for route in ROUTES:
        knobs = route["knobs"]
        is_recipe = route.get("kind") == "recipe"
        assert is_recipe or len(knobs) == 1, f"{route['id']} must change exactly one knob"
        for key, value in knobs.items():
            assert key in SCORER_KNOBS, f"{route['id']} uses unknown knob {key}"
            assert value != BASELINE_KNOBS.get(key), f"{route['id']} re-declares the baseline value"


def test_two_threshold_fusion_never_duplicates_a_filament() -> None:
    strict = [_mask([0, 1], [0, 1])]
    duplicate = _mask([0, 1], [0, 1])
    distinct = _mask([5, 6], [5, 6])
    fused = _fuse_thresholds(strict, [duplicate, distinct], 0.1)
    assert len(fused) == 2
    assert any(np.array_equal(mask, distinct) for mask in fused)


def test_two_threshold_fusion_keeps_every_strict_component() -> None:
    strict = [_mask([0], [0]), _mask([7], [7])]
    fused = _fuse_thresholds(strict, [], 0.1)
    assert len(fused) == 2
