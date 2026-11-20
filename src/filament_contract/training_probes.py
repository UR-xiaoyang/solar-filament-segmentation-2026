"""Low-overhead probes for M2 training diagnostics.

The probes are deliberately independent from the training loop. A caller can
feed one observation per batch and one activation per selected module without
changing optimization behavior. They write only small JSON summaries and never
retain tensors after an observation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch


class TrainingEpochProbe:
    """Aggregate input/output/loss/gradient signals for one epoch."""

    def __init__(self, output_dir: Path, epoch: int) -> None:
        self.output_dir = Path(output_dir)
        self.epoch = int(epoch)
        self.batch_count = 0
        self.sample_count = 0
        self._input_sum = 0.0
        self._input_sq_sum = 0.0
        self._input_count = 0
        self._target_fg = 0
        self._target_count = 0
        self._fg_prob_sum = 0.0
        self._fg_prob_count = 0
        self._bg_prob_sum = 0.0
        self._bg_prob_count = 0
        self._uncertain_count = 0
        self._output_count = 0
        self._loss_sums: dict[str, float] = {}
        self._gradient_norms: list[float] = []
        self._update_norms: list[float] = []

    @staticmethod
    def _scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().cpu())
        return float(value)

    def observe_batch(
        self,
        *,
        image: torch.Tensor,
        target: torch.Tensor,
        logits: torch.Tensor,
        loss_parts: dict[str, Any],
        grad_norm: float | None = None,
        update_norm: float | None = None,
    ) -> None:
        with torch.no_grad():
            image_cpu = image.detach().float()
            target_cpu = target.detach().bool()
            probability = torch.sigmoid(logits.detach().float())
            fg = target_cpu
            bg = ~fg
            self.batch_count += 1
            self.sample_count += int(image.shape[0])
            self._input_sum += float(image_cpu.sum().cpu())
            self._input_sq_sum += float((image_cpu * image_cpu).sum().cpu())
            self._input_count += image_cpu.numel()
            self._target_fg += int(fg.sum().cpu())
            self._target_count += target_cpu.numel()
            self._fg_prob_sum += float(probability[fg].sum().cpu()) if fg.any() else 0.0
            self._fg_prob_count += int(fg.sum().cpu())
            self._bg_prob_sum += float(probability[bg].sum().cpu()) if bg.any() else 0.0
            self._bg_prob_count += int(bg.sum().cpu())
            self._uncertain_count += int(((probability > 0.4) & (probability < 0.6)).sum().cpu())
            self._output_count += probability.numel()
            for name, value in loss_parts.items():
                self._loss_sums[name] = self._loss_sums.get(name, 0.0) + self._scalar(value)
            if grad_norm is not None:
                self._gradient_norms.append(float(grad_norm))
            if update_norm is not None:
                self._update_norms.append(float(update_norm))

    def write_epoch_summary(self) -> dict[str, Any]:
        if self.batch_count == 0:
            raise RuntimeError("cannot write an empty training probe")
        image_mean = self._input_sum / max(1, self._input_count)
        variance = self._input_sq_sum / max(1, self._input_count) - image_mean**2
        fg_mean = self._fg_prob_sum / max(1, self._fg_prob_count)
        bg_mean = self._bg_prob_sum / max(1, self._bg_prob_count)
        summary = {
            "epoch": self.epoch,
            "batch_count": self.batch_count,
            "sample_count": self.sample_count,
            "data": {
                "image_mean": image_mean,
                "image_std": max(0.0, variance) ** 0.5,
                "foreground_fraction": self._target_fg / max(1, self._target_count),
            },
            "outputs": {
                "foreground_probability_mean": fg_mean,
                "background_probability_mean": bg_mean,
                "probability_separation": fg_mean - bg_mean,
                "uncertain_pixel_fraction": self._uncertain_count / max(1, self._output_count),
            },
            "loss": {name: value / self.batch_count for name, value in self._loss_sums.items()},
            "optimization": {
                "gradient_norm": sum(self._gradient_norms) / len(self._gradient_norms) if self._gradient_norms else None,
                "update_norm": sum(self._update_norms) / len(self._update_norms) if self._update_norms else None,
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "training_probes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        return summary


class ActivationProbe:
    """Forward hooks for a selected set of modules."""

    def __init__(self, model: torch.nn.Module, include_names: set[str] | None = None) -> None:
        self.include_names = include_names
        self._stats: dict[str, list[dict[str, float | int]]] = {}
        self._handles = []
        for name, module in model.named_modules():
            if name and (include_names is None or name in include_names):
                self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                tensor = output.detach().float()
                channel_dims = tuple(i for i in range(tensor.ndim) if i != 1)
                channel_mean = tensor.mean(dim=channel_dims)
                item = {
                    "mean": float(tensor.mean().cpu()),
                    "std": float(tensor.std().cpu()),
                    "sparsity": float((tensor.abs() < 1e-6).float().mean().cpu()),
                    "dead_channels": int((channel_mean.abs() < 1e-6).sum().cpu()),
                    "channel_count": int(tensor.shape[1]) if tensor.ndim > 1 else 1,
                }
                self._stats.setdefault(name, []).append(item)
        return hook

    def summarize_and_clear(self) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for name, observations in self._stats.items():
            summary[name] = {
                "observation_count": len(observations),
                "channel_count": observations[-1]["channel_count"],
                "mean": sum(float(x["mean"]) for x in observations) / len(observations),
                "std": sum(float(x["std"]) for x in observations) / len(observations),
                "sparsity": sum(float(x["sparsity"]) for x in observations) / len(observations),
                "dead_channels": sum(int(x["dead_channels"]) for x in observations) / len(observations),
            }
        self._stats.clear()
        return summary

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
