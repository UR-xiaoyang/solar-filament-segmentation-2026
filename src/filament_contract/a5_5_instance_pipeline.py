"""A5-5 instance-aware training pipeline.

This is the orchestration boundary for A5-5: load validated config and
fold, build datasets, build the Mask R-CNN model, train, write
checkpoints and provenance, then run a final development-OOF evaluation
through the strict PQ contract.

Single-variable discipline (per the handoff document):

* The Mask R-CNN architecture is the only change vs. Control-768.
* Training protocol, fold splits, augmentation, evaluation resolution,
  post-processing thresholds, and contract hash are all frozen.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, List

import numpy as np
from PIL import Image

from .a5_5_instance_config import (
    A5_5InstanceConfig,
    config_from_mapping,
    read_config,
    write_config,
)
from .a5_5_instance_data import (
    ANNOTATION_FILE_NAME,
    InstanceSegmentationDataset,
)
from .a5_5_instance_metrics import (
    MaskRCNNPrediction,
    evaluate_maskrcnn_predictions,
)
from .a5_5_instance_model import (
    build_maskrcnn_filament,
    maskrcnn_loss_dict_to_total,
)
from .a5_5_filament_net import build_filament_maskrcnn
from .a5_strict_pq_contract import StrictPQContract


DEFAULT_DATA_ROOT = Path("data/raw/MAGFiLO_1.0_Kaggle_2026")
DEFAULT_MANIFEST = Path("artifacts/m1/split_manifest.csv")


# ---------------------------------------------------------------------------
# Helpers (mirroring m2_pipeline style).
# ---------------------------------------------------------------------------


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, values: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(values), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _append_jsonl(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(dict(values), ensure_ascii=False, sort_keys=True) + "\n")


def _atomic_torch_save(torch: Any, path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _select_device(torch: Any, requested: str) -> Any:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested but CUDA is unavailable"
        )
    return device


# ---------------------------------------------------------------------------
# DataLoader with variable-sized image collation.
# ---------------------------------------------------------------------------


def _maskrcnn_collate(batch):
    """Collate a list of (image_tensor, target_dict, image_id) into a Mask R-CNN batch.

    Mask R-CNN requires a list of image tensors and a list of target
    dicts (not a stacked batch), because images can have arbitrary shapes
    and different numbers of instances.
    """
    import torch

    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    image_ids = [item[2] for item in batch]
    return images, targets, image_ids


# ---------------------------------------------------------------------------
# One training step / one validation step.
# ---------------------------------------------------------------------------


def _train_one_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    config: A5_5InstanceConfig,
    device: Any,
    *,
    max_steps: int | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    sample_count = 0
    for step, (images, targets, _) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = [img.to(device=device, dtype=torch.float32, non_blocking=True) for img in images]
        targets = [
            {k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v for k, v in t.items()}
            for t in targets
        ]
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated() / 1024**3
            res = torch.cuda.memory_reserved() / 1024**3
            print(f"  pre-step {step}: alloc={alloc:.2f}GB reserved={res:.2f}GB", flush=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=config.amp and device.type == "cuda",
        ):
            loss_dict = model(images, targets)
            loss = maskrcnn_loss_dict_to_total(loss_dict)
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated() / 1024**3
            res = torch.cuda.memory_reserved() / 1024**3
            print(f"  post-fwd {step}: alloc={alloc:.2f}GB reserved={res:.2f}GB", flush=True)
        scaler.scale(loss).backward()
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated() / 1024**3
            res = torch.cuda.memory_reserved() / 1024**3
            print(f"  post-bwd {step}: alloc={alloc:.2f}GB reserved={res:.2f}GB", flush=True)
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(images)
        sample_count += batch_size
        for key, value in loss_dict.items():
            totals[key] += float(value.detach()) * batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        # Release the per-step loss graph and CUDA caching allocator
        # allocations. Mask R-CNN allocates many short-lived intermediates
        # whose allocator slots otherwise persist across steps and
        # trigger OOM on tight 24 GiB GPUs.
        del loss_dict, loss, images, targets
        if device.type == "cuda" and (step + 1) % 1 == 0:
            torch.cuda.empty_cache()
    if sample_count == 0:
        raise RuntimeError("training loader produced no samples")
    return {key: value / sample_count for key, value in totals.items()}


def _validate_one_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    config: A5_5InstanceConfig,
    device: Any,
    *,
    max_steps: int | None,
) -> dict[str, float]:
    """Compute validation loss by running model in train mode on the validation set."""
    model.train()
    totals: dict[str, float] = defaultdict(float)
    sample_count = 0
    with torch.inference_mode(False):
        with torch.enable_grad():
            for step, (images, targets, _) in enumerate(loader):
                if max_steps is not None and step >= max_steps:
                    break
                images = [
                    img.to(device=device, dtype=torch.float32, non_blocking=True)
                    for img in images
                ]
                targets = [
                    {
                        k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=config.amp and device.type == "cuda",
                ):
                    loss_dict = model(images, targets)
                    loss = maskrcnn_loss_dict_to_total(loss_dict)
                batch_size = len(images)
                sample_count += batch_size
                for key, value in loss_dict.items():
                    totals[key] += float(value.detach()) * batch_size
                totals["loss"] += float(loss.detach()) * batch_size
    if sample_count == 0:
        raise RuntimeError("validation loader produced no samples")
    return {key: value / sample_count for key, value in totals.items()}


# ---------------------------------------------------------------------------
# Inference helpers for post-training strict PQ evaluation.
# ---------------------------------------------------------------------------


def _run_inference(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
) -> List[MaskRCNNPrediction]:
    """Run the model in eval mode on the loader, returning per-image predictions."""
    model.eval()
    predictions: List[MaskRCNNPrediction] = []
    with torch.inference_mode():
        for images, _, image_ids in loader:
            images = [img.to(device=device, dtype=torch.float32, non_blocking=True) for img in images]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=True and device.type == "cuda",
            ):
                outputs = model(images)
            for image_id, output in zip(image_ids, outputs, strict=True):
                # Keep sigmoid float32 — mask_threshold is applied downstream.
                # Earlier ``astype(np.uint8)`` truncated soft probabilities to
                # 0/1 at the boundary, which threw away almost every pixel
                # when sigmoid values clustered below 0.5 and caused
                # strict-PQ to report zero true positives. Also squeeze the
                # channel dim so downstream resize expects a 2D mask.
                masks_np = np.asarray(output["masks"].cpu().numpy(), dtype=np.float32)
                if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np.squeeze(1)
                predictions.append(
                    MaskRCNNPrediction(
                        image_id=image_id,
                        scores=np.asarray(output["scores"].cpu().numpy(), dtype=np.float32),
                        boxes=np.asarray(output["boxes"].cpu().numpy(), dtype=np.float32),
                        masks=masks_np,
                        model_grid_hw=tuple(masks_np.shape[1:]),
                    )
                )
    return predictions


def _rasterize_gt_masks_at_eval(
    dataset: InstanceSegmentationDataset,
    evaluation_resolution: tuple[int, int],
) -> dict[str, List[np.ndarray]]:
    """Rasterize ground-truth COCO polygons to (H, W) bool masks at evaluation resolution.

    Each physical image may have multiple annotation records (one per
    annotator). Following the M2 convention used by the semantic
    pipeline, we **merge** all records for one image into one instance
    pool for training. For evaluation we still use the merged pool so
    that the resulting PQ is directly comparable to the strict PQ
    contract evaluation surface.
    """
    from .a5_5_instance_data import polygon_to_mask
    from scipy.ndimage import zoom

    eval_h, eval_w = evaluation_resolution
    out: dict[str, List[np.ndarray]] = {}
    for item in dataset._items:  # noqa: SLF001
        # Resolve all annotation records that reference this physical image.
        record_ids = dataset._index.physical_image_to_record_ids.get(  # noqa: SLF001
            item.image_id, []
        )
        masks: List[np.ndarray] = []
        for record_id in record_ids:
            ann = dataset._index.record(record_id)  # noqa: SLF001
            for poly in ann.polygons:
                mask_at_native = polygon_to_mask(poly, ann.height, ann.width)
                if mask_at_native.shape != evaluation_resolution:
                    zoom_h = eval_h / ann.height
                    zoom_w = eval_w / ann.width
                    resized = zoom(mask_at_native.astype(np.float32), (zoom_h, zoom_w), order=1)
                    mask = (resized > 0.5).astype(bool)
                else:
                    mask = mask_at_native.astype(bool)
                masks.append(mask)
        out[item.image_id] = masks
    return out


# ---------------------------------------------------------------------------
# Public entry point: train_instance_baseline.
# ---------------------------------------------------------------------------


def train_instance_baseline(
    config: A5_5InstanceConfig,
    *,
    data_root: Path,
    split_manifest: Path,
    run_dir: Path,
    device_name: str,
    smoke: bool | None = None,
) -> dict[str, Any]:
    """Train an A5-5 instance-aware baseline.

    Args:
        config: Validated A5-5 configuration.
        data_root: Path to the MAGFiLO data root containing ``train/``.
        split_manifest: Path to the frozen M1 split manifest.
        run_dir: Output directory for checkpoints and provenance.
        device_name: ``cuda:0`` etc.
        smoke: If True, override config to a 2-step smoke run.

    Returns:
        A dict of run metadata suitable for serialization to JSON.
    """
    import torch

    # Patch torchvision's roi_align to force a positive sampling ratio when
    # the input is at image-scale (e.g., GT masks at HxW = 512x512). The
    # default ``sampling_ratio=0`` allocates a [K, C, PH, PW, H, W] tensor
    # which blows up to tens of GB for typical Mask R-CNN inputs. The
    # patch below caps the sample grid at 2x2 — sufficient for the loss
    # path and never used for the inference path (which always has
    # sampling_ratio set explicitly).
    from torchvision.ops import roi_align as _orig_roi_align

    def _patched_roi_align(input, boxes, output_size, spatial_scale=1.0,
                           sampling_ratio=0, aligned=False):
        # If the input is at image scale (height/width >> output_size),
        # adaptive sampling would blow up memory. Force 2x2 sampling.
        if sampling_ratio <= 0 and input.dim() == 4:
            in_h = int(input.shape[-2])
            in_w = int(input.shape[-1])
            out_h, out_w = output_size if isinstance(output_size, (tuple, list)) else (output_size, output_size)
            if in_h > out_h * 4 or in_w > out_w * 4:
                sampling_ratio = 2
        return _orig_roi_align(input, boxes, output_size, spatial_scale,
                               sampling_ratio, aligned)

    import torchvision.models.detection.roi_heads as _rh_mod
    _rh_mod.roi_align = _patched_roi_align

    config.validate()
    effective = config
    if smoke is True or (smoke is None and config.smoke):
        effective = effective.with_overrides(
            epochs=2,
            batch_size=min(effective.batch_size, 2),
            num_workers=0,
        )
    if effective.epochs < 1:
        raise ValueError("A5-5 smoke overrides require epochs >= 1")

    device = _select_device(torch, device_name)
    _seed_everything(torch, effective.seed)

    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir / "config.json", effective)
    _write_json(
        run_dir / "run_state.json",
        {"status": "running", "run_kind": "smoke" if effective.smoke else "fold_baseline"},
    )
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    annotation_file = data_root / "train" / ANNOTATION_FILE_NAME
    if not annotation_file.is_file():
        raise FileNotFoundError(f"annotation JSON not found: {annotation_file}")

    train_dataset = InstanceSegmentationDataset(
        data_root=data_root,
        split_manifest_path=split_manifest,
        annotation_file=annotation_file,
        split="train",
        fold=effective.fold,
        image_size=effective.image_size,
        augment_d4=effective.augment_d4,
        train=True,
    )
    val_dataset = InstanceSegmentationDataset(
        data_root=data_root,
        split_manifest_path=split_manifest,
        annotation_file=annotation_file,
        split="val",
        fold=effective.fold,
        image_size=effective.image_size,
        augment_d4=False,
        train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=effective.batch_size,
        shuffle=True,
        num_workers=effective.num_workers,
        collate_fn=_maskrcnn_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=effective.num_workers > 0,
        generator=torch.Generator().manual_seed(effective.seed),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Mask R-CNN inference needs images at their native shape
        shuffle=False,
        num_workers=effective.num_workers,
        collate_fn=_maskrcnn_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=effective.num_workers > 0,
    )

    model_factory = build_filament_maskrcnn if effective.use_filament_net else build_maskrcnn_filament
    model = model_factory(effective).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective.learning_rate,
        weight_decay=effective.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=effective.epochs
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=effective.amp and device.type == "cuda"
    )

    started = time.perf_counter()
    best_loss = float("inf")
    best_epoch = -1
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"
    max_steps = 2 if effective.smoke else None
    # Smoke probes only need to confirm forward/backward pass works; full
    # validation OOMs on 24 GiB hardware when mask proposals balloon the
    # activation graph. Skip validation in smoke mode and reuse the
    # training loss as the proxy for ``best_loss``.
    skip_validation = effective.smoke

    for epoch in range(effective.epochs):
        epoch_started = time.perf_counter()
        train_metrics = _train_one_epoch(
            torch, model, train_loader, optimizer, scaler, effective, device,
            max_steps=max_steps,
        )
        if skip_validation:
            val_metrics = {"loss": float("nan"), "skipped": True}
        else:
            val_metrics = _validate_one_epoch(
                torch, model, val_loader, effective, device,
                max_steps=max_steps,
            )
        scheduler.step()
        epoch_record = {
            "epoch": epoch + 1,
            "duration_seconds": time.perf_counter() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        _append_jsonl(history_path, epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)

        checkpoint = {
            "schema_version": 1,
            "epoch": epoch + 1,
            "config": effective.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "validation": val_metrics,
            "split_manifest_sha256": _sha256(split_manifest),
            "annotation_sha256": train_dataset._index.annotation_sha256,  # noqa: SLF001
            "train_sample_count": len(train_dataset),
            "validation_sample_count": len(val_dataset),
            "run_kind": "smoke" if effective.smoke else "fold_baseline",
        }
        _atomic_torch_save(torch, last_path, checkpoint)
        # In smoke mode validation is skipped, so fall back to the
        # training loss for ``best_loss`` selection. NaN is filtered so
        # the always-false comparison does not corrupt ``best_loss``.
        candidate = float(val_metrics.get("loss", float("nan")))
        if math.isnan(candidate):
            candidate = float(train_metrics.get("loss", float("inf")))
        if math.isfinite(candidate) and candidate < best_loss:
            best_loss = candidate
            best_epoch = epoch + 1
            _atomic_torch_save(torch, best_path, checkpoint)

    # Final: load best, run inference, compute strict PQ on validation.
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    predictions = _run_inference(torch, model, val_loader, device)
    ground_truth_by_image = _rasterize_gt_masks_at_eval(
        val_dataset, effective.evaluation_resolution
    )
    contract = StrictPQContract(
        evaluation_resolution=effective.evaluation_resolution,
        probability_threshold=effective.score_threshold,
        min_instance_area=effective.min_instance_area,
        connectivity=8,
        max_instances=64,
        match_iou_threshold=effective.match_iou_threshold,
    )
    per_image_metrics = evaluate_maskrcnn_predictions(
        predictions=predictions,
        ground_truth_by_image=ground_truth_by_image,
        contract=contract,
    )

    del train_loader, val_loader
    gc.collect()

    pq_like_values = [m.pq_like for m, _ in per_image_metrics]
    rq_values = [m.rq for m, _ in per_image_metrics]
    sq_values = [m.sq for m, _ in per_image_metrics if m.sq is not None]
    total_tp = sum(m.tp_count for m, _ in per_image_metrics)
    total_fp = sum(m.fp_count for m, _ in per_image_metrics)
    total_fn = sum(m.fn_count for m, _ in per_image_metrics)

    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "run_kind": "smoke" if effective.smoke else "fold_baseline",
        "status": "completed",
        "fold": effective.fold,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "elapsed_seconds": elapsed,
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(val_dataset),
        "split_manifest_sha256": _sha256(split_manifest),
        "annotation_sha256": train_dataset._index.annotation_sha256,  # noqa: SLF001
        "config_sha256": _sha256(run_dir / "config.json"),
        "checkpoint_best_sha256": _sha256(best_path),
        "strict_pq": {
            "metric_scope": "a5_5_instance_development_oof_strict_pq_like",
            "image_count": len(per_image_metrics),
            "pq_like_mean": float(np.mean(pq_like_values)) if pq_like_values else 0.0,
            "sq_mean": float(np.mean(sq_values)) if sq_values else None,
            "rq_mean": float(np.mean(rq_values)) if rq_values else 0.0,
            "total_tp": int(total_tp),
            "total_fp": int(total_fp),
            "total_fn": int(total_fn),
            "score_threshold": effective.score_threshold,
            "match_iou_threshold": effective.match_iou_threshold,
            "evaluation_resolution": list(effective.evaluation_resolution),
        },
        "runtime": _runtime_metadata(torch, device),
    }
    _write_json(run_dir / "metrics.json", result)
    _write_json(
        run_dir / "run_state.json",
        {"status": "completed", "run_kind": result["run_kind"]},
    )
    return result


def _runtime_metadata(torch: Any, device: Any) -> dict[str, Any]:
    import PIL
    import pycocotools

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "pycocotools": getattr(pycocotools, "__version__", "unknown"),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")

    args = parser.parse_args(argv)
    config = read_config(args.config)
    result = train_instance_baseline(
        config,
        data_root=args.data_root,
        split_manifest=args.split_manifest,
        run_dir=args.run_dir,
        device_name=args.device,
        smoke=args.smoke,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MANIFEST",
    "train_instance_baseline",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())