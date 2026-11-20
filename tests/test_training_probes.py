"""Regression tests for lightweight M2 training probe telemetry."""

from __future__ import annotations

import json

import pytest


def test_epoch_probe_aggregates_training_signals_and_writes_jsonl(tmp_path):
    torch = pytest.importorskip("torch")
    from filament_contract.training_probes import TrainingEpochProbe

    probe = TrainingEpochProbe(output_dir=tmp_path, epoch=3)
    probe.observe_batch(
        image=torch.tensor([[[[0.0, 1.0], [0.5, 0.5]]]]),
        target=torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]]),
        logits=torch.tensor([[[[-4.0, 4.0], [3.0, -3.0]]]]),
        loss_parts={"bce": torch.tensor(0.25), "soft_dice": torch.tensor(0.75)},
        grad_norm=2.5,
        update_norm=0.05,
    )

    summary = probe.write_epoch_summary()

    assert summary["epoch"] == 3
    assert summary["batch_count"] == 1
    assert summary["data"]["foreground_fraction"] == pytest.approx(0.5)
    assert summary["outputs"]["foreground_probability_mean"] > 0.9
    assert summary["outputs"]["background_probability_mean"] < 0.1
    assert summary["outputs"]["probability_separation"] > 0.8
    assert summary["loss"]["bce"] == pytest.approx(0.25)
    assert summary["optimization"]["gradient_norm"] == pytest.approx(2.5)

    lines = (tmp_path / "training_probes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == summary


def test_activation_probe_captures_selected_module_stats_without_retaining_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    from filament_contract.training_probes import ActivationProbe

    model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, kernel_size=1), torch.nn.ReLU())
    probe = ActivationProbe(model, include_names={"0"})
    _ = model(torch.ones(1, 1, 2, 2))
    stats = probe.summarize_and_clear()
    probe.close()

    assert list(stats) == ["0"]
    assert stats["0"]["observation_count"] == 1
    assert stats["0"]["channel_count"] == 2
    assert "mean" in stats["0"]
