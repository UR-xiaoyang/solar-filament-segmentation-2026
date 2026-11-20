from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load(name: str, relative: str):
    path = Path(__file__).parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load("divergence_gate_v2", "scripts/divergence_gate_v2.py")
routes_module = _load("divergence100_routes", "scripts/divergence100_routes.py")


def _metrics(fold: int, count: int, dice: float, *, threshold: float = 0.5,
             min_area: int = 256, resolution: tuple[int, int] = (1024, 1024)) -> dict:
    return {
        "metric_scope": "local_leaderboard_proxy_not_official",
        "fold": fold,
        "sample_count": count,
        "primary": {"mean_dice_per_image": dice},
        "dice_distribution": {"p05": dice - 0.1},
        "relation_distribution": {
            "one_to_one": 100,
            "one_to_many": 2,
            "many_to_one": 3,
            "missed": 10,
            "spurious": 12,
        },
        "contract": {
            "stored_probability_resolution": list(resolution),
            "evaluation_resolution": 2048,
            "threshold": threshold,
            "min_instance_area": min_area,
            "match_iou_threshold": 0.5,
            "gt_universe": "all COCO records per physical image",
            "probability_decode": "uint16/65535 when dtype is uint16",
        },
    }


def _write_fixture(root: Path, *, candidate_delta: float,
                   candidate_threshold: float = 0.5,
                   control_threshold: float = 0.5) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    candidate, control = root / "candidate", root / "control"
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=["image_id", "role", "fold"])
        out.writeheader()
        for fold, count in gate.EXPECTED_COUNTS.items():
            for index in range(count):
                out.writerow({"image_id": f"f{fold}-{index}", "role": "development", "fold": fold})
    for fold, count in gate.EXPECTED_COUNTS.items():
        for side in (candidate, control):
            (side / f"fold{fold}").mkdir(parents=True)
            oof = side / f"fold{fold}" / "oof_probabilities"
            oof.mkdir()
            for index in range(count):
                (oof / f"f{fold}-{index}.npy").touch()
        (candidate / f"fold{fold}" / "metrics.json").write_text(json.dumps(
            _metrics(fold, count, 0.70 + candidate_delta, threshold=candidate_threshold)))
        (control / f"fold{fold}" / "metrics.json").write_text(json.dumps(
            _metrics(fold, count, 0.70, threshold=control_threshold)))
    return candidate, control, manifest


def _args(candidate: Path, control: Path, manifest: Path, out: Path, mode: str = "full"):
    return gate.argparse.Namespace(
        candidate=candidate, control=control, manifest=manifest, out=out, mode=mode,
        candidate_oof_root=None, control_oof_root=None,
        candidate_oof_template="fold{fold}/oof_probabilities",
        control_oof_template="fold{fold}/oof_probabilities",
        candidate_metrics_template="fold{fold}/metrics.json",
        control_metrics_template="fold{fold}/metrics.json",
        allow_contract_diff=[],
        fold_max_drop=0.01, overall_gain=0.01, min_same_direction_folds=4,
        pilot_min_delta=0.005)


def test_gate_passes_complete_paired_improvement(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(tmp_path, candidate_delta=0.02)
    code, report = gate.evaluate(_args(candidate, control, manifest, tmp_path / "report.json"))
    assert code == 0
    assert report["status"] == "passed"
    assert report["coverage"]["total_images"] == 618
    assert report["checks"] == {
        "g1_fold_floor": True,
        "g2_overall_gain": True,
        "g3_direction": True,
        "g4_complete_pair": True,
    }


def test_gate_rejects_complete_paired_regression(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(tmp_path, candidate_delta=-0.02)
    code, report = gate.evaluate(_args(candidate, control, manifest, tmp_path / "report.json"))
    assert code == 1
    assert report["status"] == "no_go"
    assert report["checks"]["g1_fold_floor"] is False


def test_gate_marks_undeclared_contract_difference_not_evaluable(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(
        tmp_path, candidate_delta=0.02, candidate_threshold=0.55, control_threshold=0.5)
    code, report = gate.evaluate(_args(candidate, control, manifest, tmp_path / "report.json"))
    assert code == 2
    assert report["status"] == "not_evaluable"
    assert any("undeclared contract difference" in error for error in report["errors"])


def test_gate_allows_declared_threshold_difference(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(
        tmp_path, candidate_delta=0.02, candidate_threshold=0.45, control_threshold=0.5)
    args = _args(candidate, control, manifest, tmp_path / "report.json")
    args.allow_contract_diff = ["threshold"]
    code, report = gate.evaluate(args)
    assert code == 0
    assert report["status"] == "passed"
    assert report["folds"][0]["declared_contract_diff"]["threshold"]["candidate"] == 0.45


def test_gate_marks_fixed_contract_difference_not_evaluable(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(tmp_path, candidate_delta=0.02)
    path = candidate / "fold1" / "metrics.json"
    payload = json.loads(path.read_text())
    payload["contract"]["match_iou_threshold"] = 0.6
    path.write_text(json.dumps(payload))
    code, report = gate.evaluate(_args(candidate, control, manifest, tmp_path / "report.json"))
    assert code == 2
    assert report["status"] == "not_evaluable"
    assert any("fixed scoring contract differs" in error for error in report["errors"])


def test_gate_marks_missing_oof_coverage_not_evaluable(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(tmp_path, candidate_delta=0.02)
    (candidate / "fold3" / "oof_probabilities" / "f3-123.npy").unlink()
    code, report = gate.evaluate(_args(candidate, control, manifest, tmp_path / "report.json"))
    assert code == 2
    assert report["status"] == "not_evaluable"
    assert any("candidate: fold 3: missing 1 OOF predictions" in error for error in report["errors"])


def test_gate_accepts_recorded_hashes_when_probabilities_purged(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(tmp_path, candidate_delta=0.02)
    for fold in gate.FOLDS:
        ids = [f"f{fold}-{index}" for index in range(gate.EXPECTED_COUNTS[fold])]
        for side in (candidate, control):
            path = side / f"fold{fold}" / "metrics.json"
            payload = json.loads(path.read_text())
            payload["oof_probability_hashes"] = {image_id: "0" * 64 for image_id in ids}
            path.write_text(json.dumps(payload))
            for index in range(gate.EXPECTED_COUNTS[fold]):
                (side / f"fold{fold}" / "oof_probabilities" / f"f{fold}-{index}.npy").unlink()
            (side / f"fold{fold}" / "oof_probabilities").rmdir()
    code, report = gate.evaluate(_args(candidate, control, manifest, tmp_path / "report.json"))
    assert code == 0
    assert report["status"] == "passed"


def test_gate_rejects_purged_probabilities_without_hash_evidence(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(tmp_path, candidate_delta=0.02)
    for fold in gate.FOLDS:
        for side in (candidate, control):
            for index in range(gate.EXPECTED_COUNTS[fold]):
                (side / f"fold{fold}" / "oof_probabilities" / f"f{fold}-{index}.npy").unlink()
            (side / f"fold{fold}" / "oof_probabilities").rmdir()
    code, report = gate.evaluate(_args(candidate, control, manifest, tmp_path / "report.json"))
    assert code == 2
    assert report["status"] == "not_evaluable"
    assert any("no recorded probability hashes" in error for error in report["errors"])


def test_gate_accepts_separate_metrics_and_oof_roots(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(tmp_path / "oof", candidate_delta=0.02)
    candidate_metrics, control_metrics = tmp_path / "candidate-metrics", tmp_path / "control-metrics"
    for fold in gate.FOLDS:
        for source, destination in ((candidate, candidate_metrics), (control, control_metrics)):
            target = destination / f"fold{fold}"
            target.mkdir(parents=True, exist_ok=True)
            (target / "metrics.json").write_text((source / f"fold{fold}" / "metrics.json").read_text())
    args = _args(candidate_metrics, control_metrics, manifest, tmp_path / "report.json")
    args.candidate_oof_root = candidate
    args.control_oof_root = control
    code, report = gate.evaluate(args)
    assert code == 0
    assert report["status"] == "passed"


def test_gate_pilot_mode_uses_single_fold(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(tmp_path, candidate_delta=0.02)
    code, report = gate.evaluate(
        _args(candidate, control, manifest, tmp_path / "report.json", mode="pilot"))
    assert code == 0
    assert report["status"] == "pilot_pass"
    assert report["mode"] == "pilot"
    assert len(report["folds"]) == 1


def test_gate_pilot_mode_reports_no_go_and_ambiguous(tmp_path: Path) -> None:
    candidate, control, manifest = _write_fixture(tmp_path, candidate_delta=-0.02)
    code, report = gate.evaluate(
        _args(candidate, control, manifest, tmp_path / "report.json", mode="pilot"))
    assert code == 1
    assert report["status"] == "pilot_no_go"

    candidate, control, manifest = _write_fixture(tmp_path / "second", candidate_delta=0.001)
    code, report = gate.evaluate(
        _args(candidate, control, manifest, tmp_path / "report.json", mode="pilot"))
    assert code == 1
    assert report["status"] == "pilot_ambiguous"


def test_routes_registry_has_100_distinct_mechanisms() -> None:
    routes = routes_module.ROUTES
    assert len(routes) == 100
    assert [route["id"] for route in routes] == list(range(1, 101))
    names = [route["route"] for route in routes]
    assert len(set(names)) == 100
    for route in routes:
        assert route["kind"] in ("train", "postprocess")
        if route["kind"] == "postprocess":
            assert set(route["postprocess"]) == {"threshold", "min_area"}
        else:
            assert route["config"], route["id"]
            assert set(route["config"]).issubset(set(routes_module.FROZEN_CONTROL))


def test_postprocess_routes_declare_their_contract_difference() -> None:
    for route in routes_module.ROUTES:
        if route["kind"] == "postprocess":
            assert "threshold" in route["allow_diff"]
