from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/build_project_progress_dashboard.py"
SPEC = importlib.util.spec_from_file_location("progress_dashboard", MODULE_PATH)
assert SPEC and SPEC.loader
DASHBOARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DASHBOARD)


def test_builds_standalone_offline_dashboard(tmp_path: Path) -> None:
    output = tmp_path / "dashboard.html"
    rendered = DASHBOARD.build(ROOT, ROOT / "reports/project_progress_evidence.json", output)

    assert output.read_text(encoding="utf-8") == rendered
    assert rendered.count("<h1>") == 1
    assert '<main id="content">' in rendered
    assert 'class="skip"' in rendered
    assert "M6B" in rendered
    assert "0.66" in rendered
    assert "reject_stop_before_fold1" in rendered
    assert "M8 只拒绝固定 1024/768 patch" in rendered
    assert "test ground truth" in rendered
    assert "https://" not in rendered
    assert "http://" not in rendered
    assert "<title id=\"m8-chart-title\">" in rendered
    assert "<desc id=\"m9-chart-desc\">" in rendered


def test_render_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"
    DASHBOARD.build(ROOT, ROOT / "reports/project_progress_evidence.json", first)
    DASHBOARD.build(ROOT, ROOT / "reports/project_progress_evidence.json", second)
    assert first.read_bytes() == second.read_bytes()


def test_check_detects_stale_dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "dashboard.html"
    DASHBOARD.build(ROOT, ROOT / "reports/project_progress_evidence.json", output)
    output.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_project_progress_dashboard.py",
            "--output",
            str(output),
            "--check",
        ],
    )
    with pytest.raises(DASHBOARD.EvidenceError, match="stale or missing"):
        DASHBOARD.main()


def test_rejects_inconsistent_evidence(tmp_path: Path) -> None:
    broken = tmp_path / "evidence.json"
    broken.write_text('{"title":"broken"}', encoding="utf-8")
    with pytest.raises(DASHBOARD.EvidenceError, match="missing required keys"):
        DASHBOARD.load_view_model(ROOT, broken)


def test_dashboard_does_not_make_prohibited_claims(tmp_path: Path) -> None:
    output = tmp_path / "dashboard.html"
    DASHBOARD.build(ROOT, ROOT / "reports/project_progress_evidence.json", output)
    rendered = output.read_text(encoding="utf-8")
    assert "官方本地分" not in rendered
    assert "已评估 test ground truth" not in rendered
    assert "所有多尺度方法均失败" not in rendered
