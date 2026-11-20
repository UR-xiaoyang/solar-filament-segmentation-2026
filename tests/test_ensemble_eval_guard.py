"""Leak guards for the cross-fold evaluation script.

`scripts/ensemble_tta_evaluate.py` must never score a fold with a model that was
trained on it, so its two argument guards are pinned here. The guards run before
any torch import, which is why this file works without a GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# A partial local checkout has no filament_contract.m2_config; the real run is on the GPU host.
pytest.importorskip("filament_contract.m2_config",
                    reason="pipeline modules absent from this checkout; run on the GPU host")

from ensemble_tta_evaluate import main  # noqa: E402


def run(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["ensemble_tta_evaluate.py", *argv])
    main()


def test_multi_model_oof_evaluation_is_refused(monkeypatch, tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run(monkeypatch, ["--checkpoints", "a.pt", "b.pt",
                          "--checkpoint-folds", "0", "1",
                          "--fold", "0", "--out", str(tmp_path / "out")])
    assert "leak" in str(excinfo.value).lower()


def test_cross_fold_members_are_refused_even_with_the_seed_flag(monkeypatch, tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run(monkeypatch, ["--checkpoints", "a.pt", "b.pt",
                          "--checkpoint-folds", "0", "2",
                          "--fold", "0", "--same-fold-seeds",
                          "--out", str(tmp_path / "out")])
    assert "leak" in str(excinfo.value).lower()


def test_checkpoint_must_be_the_one_validated_on_the_fold(monkeypatch, tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run(monkeypatch, ["--checkpoints", "a.pt", "--checkpoint-folds", "1",
                          "--fold", "0", "--out", str(tmp_path / "out")])
    assert "leak-free pairing" in str(excinfo.value)


def test_checkpoint_list_must_pair_with_fold_list(monkeypatch, tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run(monkeypatch, ["--checkpoints", "a.pt", "b.pt", "--checkpoint-folds", "0",
                          "--fold", "0", "--out", str(tmp_path / "out")])
    assert "one-to-one" in str(excinfo.value)
