"""Regression tests for nested judge evaluation coverage and selection."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quality_judge_nested_outer_v2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("quality_judge_nested_outer_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_manifest_keeps_zero_candidate_images():
    module = load_module()
    image_rows = [
        {"image_id": "with-candidate", "fold": "0", "candidate_count": "1"},
        {"image_id": "zero-candidate", "fold": "1", "candidate_count": "0"},
    ]
    candidates = [{"image_id": "with-candidate", "candidate_id": "0"}]

    metadata = module.build_image_metadata(image_rows, candidates)

    assert list(metadata) == ["with-candidate", "zero-candidate"]
    assert metadata["zero-candidate"] == {"fold": 1, "candidate_count": 0}


def test_aggregate_summary_uses_contract_metric_keys():
    module = load_module()

    summary = module.summarize_aggregate(
        {
            "pq_like_mean": 0.123,
            "pq_like_global": 0.098,
            "total_tp": 10,
            "total_fp": 25,
            "total_fn": 5,
        },
        image_count=2,
        zero_candidate_image_count=1,
    )

    assert summary["mean_pq_like"] == 0.123
    assert summary["global_pq_like"] == 0.098
    assert summary["fp_tp"] == 2.5
    assert summary["zero_candidate_image_count"] == 1


def test_inner_threshold_tie_uses_more_conservative_threshold():
    module = load_module()

    assert module.select_inner_threshold([
        {"threshold": 0.50, "mean_pq_like": 0.140},
        {"threshold": 0.70, "mean_pq_like": 0.140},
        {"threshold": 0.80, "mean_pq_like": 0.130},
    ]) == 0.70
