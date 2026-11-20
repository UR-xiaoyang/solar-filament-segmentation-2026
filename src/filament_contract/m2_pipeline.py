"""Train, diagnose, and submit the frozen M2 semantic segmentation baseline."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from .contract import IMAGE_SHAPE
from .m2_augmentation import RepeatedTrainingDataset
from .m2_config import M2Config, config_from_mapping, read_config, write_config
from .m2_data import (
    FROZEN_M1_SPLIT_MANIFEST_SHA256,
    CocoSemanticIndex,
    M2SegmentationDataset,
    M2TestDataset,
    build_target_cache,
)
from .m2_model import build_bce_dice_loss, build_model, prepare_model_input
from .metrics import DIAGNOSTIC_METRIC_SCOPE, diagnostic_instance_metrics
from .postprocess import instances_to_submission_records, probability_to_instances
from .submission import validate_submission_file, write_submission


DEFAULT_DATA_ROOT = Path("data/raw/MAGFiLO_1.0_Kaggle_2026")
DEFAULT_MANIFEST = Path("artifacts/m1/split_manifest.csv")
DEFAULT_CONFIG = Path("configs/m2_unetpp_resnet34_fold0.json")
DEFAULT_CACHE = Path("artifacts/m2/cache/targets")


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
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(values), ensure_ascii=False, sort_keys=True) + "\n")


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
        raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    return device


def _d4_augment(torch: Any, image: Any, target: Any) -> tuple[Any, Any]:
    augmented_images = []
    augmented_targets = []
    for sample_image, sample_target in zip(image, target, strict=True):
        rotation = int(torch.randint(0, 4, (), device=image.device).item())
        if rotation:
            sample_image = torch.rot90(sample_image, rotation, dims=(-2, -1))
            sample_target = torch.rot90(sample_target, rotation, dims=(-2, -1))
        if bool(torch.rand((), device=image.device) < 0.5):
            sample_image = torch.flip(sample_image, dims=(-1,))
            sample_target = torch.flip(sample_target, dims=(-1,))
        augmented_images.append(sample_image)
        augmented_targets.append(sample_target)
    return torch.stack(augmented_images), torch.stack(augmented_targets)


def _build_loader(
    torch: Any,
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> Any:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def _feature_dataset_options(config: M2Config) -> dict[str, object]:
    """Return the complete input-feature contract shared by every data path."""

    config.validate()
    residual_bounds = (
        None
        if config.local_contrast_residual_lower is None
        else (
            config.local_contrast_residual_lower,
            config.local_contrast_residual_upper,
        )
    )
    return {
        "input_feature_mode": config.input_feature_mode,
        "local_contrast_blur_radius": config.local_contrast_blur_radius,
        "local_contrast_percentiles": (
            config.local_contrast_percentile_low,
            config.local_contrast_percentile_high,
        ),
        "local_contrast_bounds": residual_bounds,
    }


def _feature_cache_metadata(dataset: Any) -> dict[str, object]:
    """Return immutable feature-cache provenance for run artifacts."""

    index = getattr(dataset, "feature_cache_index", None)
    if index is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "path": str(index.root),
        "manifest_sha256": index.manifest_sha256,
        "feature_contract_sha256": index.feature_contract_sha256,
        "record_count": len(index.records),
    }


def _train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    loss_function: Any,
    config: M2Config,
    device: Any,
    *,
    max_steps: int | None,
) -> dict[str, float]:
    model.train()
    totals = defaultdict(float)
    sample_count = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        if config.augment_d4:
            image, target = _d4_augment(torch, image, target)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=config.amp and device.type == "cuda",
        ):
            logits = model(prepare_model_input(image, config))
            loss, parts = loss_function(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(image.shape[0])
        sample_count += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["bce"] += float(parts["bce"]) * batch_size
        totals["soft_dice"] += float(parts["soft_dice"]) * batch_size
    if sample_count == 0:
        raise RuntimeError("training loader produced no samples")
    return {key: value / sample_count for key, value in totals.items()}


def _validate(
    torch: Any,
    model: Any,
    loader: Any,
    loss_function: Any,
    config: M2Config,
    device: Any,
    *,
    collect_probabilities: bool,
    max_steps: int | None,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    model.eval()
    loss_total = 0.0
    sample_count = 0
    macro_dice = 0.0
    macro_iou = 0.0
    intersection_total = 0
    prediction_area_total = 0
    target_area_total = 0
    probabilities: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            image = batch["image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            target = batch["target"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            binary_target = batch["binary_target"].to(
                device=device, dtype=torch.bool, non_blocking=True
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                logits = model(prepare_model_input(image, config))
                loss, _ = loss_function(logits, target)
            probability = torch.sigmoid(logits).float()
            prediction = probability >= config.probability_threshold
            axes = tuple(range(1, prediction.ndim))
            intersection = (prediction & binary_target).sum(dim=axes).cpu().numpy()
            prediction_area = prediction.sum(dim=axes).cpu().numpy()
            target_area = binary_target.sum(dim=axes).cpu().numpy()
            area_sum = prediction_area + target_area
            union = prediction_area + target_area - intersection
            dice = np.divide(
                2.0 * intersection,
                area_sum,
                out=np.ones_like(intersection, dtype=np.float64),
                where=area_sum != 0,
            )
            iou = np.divide(
                intersection,
                union,
                out=np.ones_like(intersection, dtype=np.float64),
                where=union != 0,
            )
            current_batch = int(image.shape[0])
            loss_total += float(loss) * current_batch
            sample_count += current_batch
            macro_dice += float(dice.sum())
            macro_iou += float(iou.sum())
            intersection_total += int(intersection.sum())
            prediction_area_total += int(prediction_area.sum())
            target_area_total += int(target_area.sum())
            if collect_probabilities:
                probability_numpy = probability[:, 0].cpu().numpy().astype(np.float32)
                for image_id, image_probability in zip(
                    batch["image_id"], probability_numpy, strict=True
                ):
                    if image_id in probabilities:
                        raise RuntimeError(f"duplicate validation image ID: {image_id}")
                    probabilities[image_id] = np.ascontiguousarray(image_probability)
    if sample_count == 0:
        raise RuntimeError("validation loader produced no samples")
    micro_denominator = prediction_area_total + target_area_total
    micro_union = micro_denominator - intersection_total
    metrics: dict[str, float | int | str] = {
        "metric_scope": DIAGNOSTIC_METRIC_SCOPE,
        "sample_count": sample_count,
        "loss": loss_total / sample_count,
        "dice_macro": macro_dice / sample_count,
        "iou_macro": macro_iou / sample_count,
        "dice_micro": (
            1.0 if micro_denominator == 0 else 2.0 * intersection_total / micro_denominator
        ),
        "iou_micro": 1.0 if micro_union == 0 else intersection_total / micro_union,
        "prediction_area": prediction_area_total,
        "target_area": target_area_total,
        "intersection": intersection_total,
    }
    return metrics, probabilities


def _atomic_torch_save(torch: Any, path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _save_probability(path: Path, image_id: str, probability: np.ndarray) -> Path:
    if probability.ndim != 2 or probability.dtype != np.float32:
        raise ValueError("probability artifact must be a float32 two-dimensional array")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("probability artifact values must be finite and in [0, 1]")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        image_id=np.asarray(image_id),
        probability=probability,
        model_shape=np.asarray(probability.shape, dtype=np.int32),
    )
    temporary.replace(path)
    return path


def load_probability(path: Path, *, expected_image_id: str | None = None) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            image_id = str(artifact["image_id"])
            probability = np.asarray(artifact["probability"], dtype=np.float32).copy()
            model_shape = tuple(int(value) for value in artifact["model_shape"])
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid probability artifact: {path}") from error
    if expected_image_id is not None and image_id != expected_image_id:
        raise ValueError(f"probability image ID mismatch: expected {expected_image_id}, got {image_id}")
    if probability.ndim != 2 or probability.shape != model_shape:
        raise ValueError(f"probability shape contract mismatch: {path}")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError(f"probability values fall outside [0, 1]: {path}")
    return probability


def _resize_probability(probability: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    resized = Image.fromarray(probability, mode="F").resize(
        (width, height), resample=Image.Resampling.BILINEAR
    )
    return np.ascontiguousarray(np.asarray(resized), dtype=np.float32)


def _write_probability_directory(
    output_dir: Path, probabilities: Mapping[str, np.ndarray]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for image_id in sorted(probabilities):
        path = _save_probability(output_dir / f"{image_id}.npz", image_id, probabilities[image_id])
        hashes[image_id] = _sha256(path)
    return hashes


def _rasterize_record_instances(
    index: CocoSemanticIndex, record_id: str, shape: tuple[int, int]
) -> list[np.ndarray]:
    record = index.records_by_id[record_id]
    target_height, target_width = shape
    x_scale = target_width / record.width
    y_scale = target_height / record.height
    masks: list[np.ndarray] = []
    for polygon in record.polygons:
        scaled = [
            coordinate * (x_scale if offset % 2 == 0 else y_scale)
            for offset, coordinate in enumerate(polygon)
        ]
        rle = mask_utils.frPyObjects([scaled], target_height, target_width)[0]
        masks.append(np.ascontiguousarray(mask_utils.decode(rle) != 0, dtype=np.uint8))
    return masks


def _write_oof_instance_diagnostics(
    output_path: Path,
    probabilities: Mapping[str, np.ndarray],
    index: CocoSemanticIndex,
    config: M2Config,
) -> dict[str, Any]:
    columns = (
        "image_id",
        "annotation_record_id",
        "prediction_count",
        "target_count",
        "matched_count",
        "false_positive_count",
        "false_negative_count",
        "absolute_count_error",
        "merged_prediction_count",
        "split_target_count",
        "mean_matched_dice",
        "mean_matched_iou",
    )
    rows: list[dict[str, object]] = []
    image_values: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for image_id in sorted(probabilities):
        probability = _resize_probability(probabilities[image_id], IMAGE_SHAPE)
        predictions = probability_to_instances(
            probability,
            threshold=config.probability_threshold,
            min_area=config.min_instance_area,
            connectivity=config.connectivity,
            max_instances=config.max_instances_per_image,
        )
        for record_id in index.record_ids_by_image[image_id]:
            targets = _rasterize_record_instances(index, record_id, probability.shape)
            metrics = diagnostic_instance_metrics(predictions, targets)
            row = {
                "image_id": image_id,
                "annotation_record_id": record_id,
                "prediction_count": metrics.prediction_count,
                "target_count": metrics.target_count,
                "matched_count": metrics.matched_count,
                "false_positive_count": metrics.false_positive_count,
                "false_negative_count": metrics.false_negative_count,
                "absolute_count_error": metrics.absolute_count_error,
                "merged_prediction_count": metrics.merged_prediction_count,
                "split_target_count": metrics.split_target_count,
                "mean_matched_dice": (
                    "" if metrics.mean_matched_dice is None else metrics.mean_matched_dice
                ),
                "mean_matched_iou": (
                    "" if metrics.mean_matched_iou is None else metrics.mean_matched_iou
                ),
            }
            rows.append(row)
            for field in (
                "prediction_count",
                "target_count",
                "matched_count",
                "false_positive_count",
                "false_negative_count",
                "absolute_count_error",
                "merged_prediction_count",
                "split_target_count",
            ):
                image_values[image_id][field].append(float(row[field]))
            if metrics.mean_matched_dice is not None:
                image_values[image_id]["mean_matched_dice"].append(metrics.mean_matched_dice)
            if metrics.mean_matched_iou is not None:
                image_values[image_id]["mean_matched_iou"].append(metrics.mean_matched_iou)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {
        "metric_scope": DIAGNOSTIC_METRIC_SCOPE,
        "matching_policy": "greedy_descending_iou",
        "match_iou_threshold": 0.5,
        "relation_iou_threshold": 0.0,
        "aggregation": "mean_per_annotation_record_then_macro_mean_over_physical_images",
        "physical_image_count": len(image_values),
        "annotation_record_count": len(rows),
    }
    fields = sorted({field for values in image_values.values() for field in values})
    for field in fields:
        per_image = [
            float(np.mean(values[field]))
            for values in image_values.values()
            if values[field]
        ]
        summary[f"{field}_macro"] = float(np.mean(per_image)) if per_image else None
    return summary


def _write_instance_outputs(
    probability_dir: Path,
    image_ids: Sequence[str],
    output_csv: Path,
    instance_manifest_path: Path,
    config: M2Config,
) -> dict[str, Any]:
    records = []
    manifest_rows: list[dict[str, object]] = []
    zero_prediction_images: list[str] = []
    for image_id in sorted(image_ids):
        model_probability = load_probability(
            probability_dir / f"{image_id}.npz", expected_image_id=image_id
        )
        full_probability = _resize_probability(model_probability, IMAGE_SHAPE)
        instances = probability_to_instances(
            full_probability,
            threshold=config.probability_threshold,
            min_area=config.min_instance_area,
            connectivity=config.connectivity,
            max_instances=config.max_instances_per_image,
        )
        if not instances:
            zero_prediction_images.append(image_id)
        image_records = instances_to_submission_records(image_id, instances)
        records.extend(image_records)
        for instance, record in zip(instances, image_records, strict=True):
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "filament_id": record.filament_id,
                    "instance_id": instance.instance_id,
                    "area": instance.area,
                    "score": f"{instance.score:.8f}",
                }
            )
    write_submission(output_csv, records)
    validate_submission_file(output_csv, set(image_ids))
    instance_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with instance_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "filament_id", "instance_id", "area", "score"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "image_count": len(image_ids),
        "instance_count": len(records),
        "zero_prediction_image_count": len(zero_prediction_images),
        "zero_prediction_images": zero_prediction_images,
        "submission_sha256": _sha256(output_csv),
        "instance_manifest_sha256": _sha256(instance_manifest_path),
    }


def _runtime_metadata(torch: Any, device: Any) -> dict[str, Any]:
    import PIL
    import pycocotools
    import segmentation_models_pytorch as smp

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "pycocotools": getattr(pycocotools, "__version__", "unknown"),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "segmentation_models_pytorch": smp.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def train_baseline(
    config: M2Config,
    *,
    data_root: Path,
    split_manifest: Path,
    run_dir: Path,
    target_cache_root: Path,
    device_name: str,
    smoke: bool = False,
    skip_cache: bool = False,
    feature_cache_dir: Path | None = None,
    train_repeat_factor: int | None = None,
    init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    import torch

    config.validate()
    effective = (
        config.with_overrides(train_repeat_factor=train_repeat_factor)
        if train_repeat_factor is not None
        else config
    )
    if smoke:
        effective = effective.with_overrides(
            epochs=1,
            batch_size=min(effective.batch_size, 2),
            num_workers=0,
        )
    device = _select_device(torch, device_name)
    _seed_everything(torch, effective.seed)
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir / "config.json", effective)
    _write_json(
        run_dir / "run_state.json",
        {"status": "running", "run_kind": "smoke" if smoke else "fold_baseline"},
    )
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    dataset_options = {
        "data_root": data_root,
        "split_manifest": split_manifest,
        "fold": effective.fold,
        "image_size": effective.image_size,
        "target_mode": effective.target_mode,
        "normalization_percentiles": (
            effective.percentile_low,
            effective.percentile_high,
        ),
        "target_cache_root": target_cache_root,
        "expected_manifest_sha256": FROZEN_M1_SPLIT_MANIFEST_SHA256,
        "feature_cache_dir": feature_cache_dir,
        **_feature_dataset_options(effective),
        "config": asdict(effective),
    }
    train_base_dataset = M2SegmentationDataset(split="train", **dataset_options)
    val_base_dataset = M2SegmentationDataset(split="val", **dataset_options)
    feature_cache = _feature_cache_metadata(train_base_dataset)
    if feature_cache != _feature_cache_metadata(val_base_dataset):
        raise RuntimeError("train and validation feature-cache provenance differs")
    if not smoke and not skip_cache:
        train_base_dataset.build_target_cache()
        val_base_dataset.build_target_cache()
    base_train_sample_count = len(train_base_dataset)
    train_dataset = RepeatedTrainingDataset(
        train_base_dataset,
        repeat_factor=effective.train_repeat_factor,
    )
    val_dataset = val_base_dataset
    if smoke:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(8, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(4, len(val_dataset))))
    train_loader = _build_loader(
        torch,
        train_dataset,
        batch_size=effective.batch_size,
        num_workers=effective.num_workers,
        shuffle=True,
        seed=effective.seed,
    )
    val_loader = _build_loader(
        torch,
        val_dataset,
        batch_size=effective.batch_size,
        num_workers=effective.num_workers,
        shuffle=False,
        seed=effective.seed,
    )

    model = build_model(effective).to(device)
    initialization_metadata: dict[str, Any] = {"path": None, "sha256": None}
    if init_checkpoint is not None:
        init_checkpoint = Path(init_checkpoint).resolve()
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"initialization checkpoint does not exist: {init_checkpoint}")
        initialization = torch.load(init_checkpoint, map_location=device, weights_only=False)
        if not isinstance(initialization, dict) or "model_state_dict" not in initialization:
            raise ValueError("initialization checkpoint lacks model_state_dict")
        model.load_state_dict(initialization["model_state_dict"], strict=True)
        initialization_metadata = {
            "path": str(init_checkpoint),
            "sha256": _sha256(init_checkpoint),
        }
    loss_function = build_bce_dice_loss(effective).to(device)
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_dice = -1.0
    best_epoch = -1
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"
    max_steps = 2 if smoke else None
    for epoch in range(effective.epochs):
        epoch_started = time.perf_counter()
        training = _train_epoch(
            torch,
            model,
            train_loader,
            optimizer,
            scaler,
            loss_function,
            effective,
            device,
            max_steps=max_steps,
        )
        validation, _ = _validate(
            torch,
            model,
            val_loader,
            loss_function,
            effective,
            device,
            collect_probabilities=False,
            max_steps=max_steps,
        )
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_record = {
            "epoch": epoch + 1,
            "duration_seconds": time.perf_counter() - epoch_started,
            "learning_rate": epoch_learning_rate,
            "train": training,
            "validation": validation,
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
            "validation": validation,
            "split_manifest_sha256": _sha256(split_manifest),
            "annotation_sha256": train_base_dataset.annotation_index.annotation_sha256,
            "base_train_sample_count": base_train_sample_count,
            "train_repeat_factor": effective.train_repeat_factor,
            "train_sample_count": len(train_dataset),
            "feature_cache": feature_cache,
            "run_kind": "smoke" if smoke else "fold_baseline",
            "initialization_checkpoint": initialization_metadata,
        }
        _atomic_torch_save(torch, last_path, checkpoint)
        current_dice = float(validation["dice_macro"])
        if current_dice > best_dice:
            best_dice = current_dice
            best_epoch = epoch + 1
            _atomic_torch_save(torch, best_path, checkpoint)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_validation, probabilities = _validate(
        torch,
        model,
        val_loader,
        loss_function,
        effective,
        device,
        collect_probabilities=True,
        max_steps=max_steps,
    )
    del train_loader, val_loader
    gc.collect()
    probability_dir = run_dir / "oof_probabilities"
    probability_hashes = _write_probability_directory(probability_dir, probabilities)
    annotation_index = val_base_dataset.annotation_index
    instance_diagnostics = _write_oof_instance_diagnostics(
        run_dir / "oof_instance_diagnostics.csv",
        probabilities,
        annotation_index,
        effective,
    )
    oof_instances = _write_instance_outputs(
        probability_dir,
        sorted(probabilities),
        run_dir / "oof_instances.csv",
        run_dir / "oof_instance_manifest.csv",
        effective,
    )
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "run_kind": "smoke" if smoke else "fold_baseline",
        "status": "completed",
        "fold": effective.fold,
        "base_train_sample_count": base_train_sample_count,
        "train_sample_count": len(train_dataset),
        "train_repeat_factor": effective.train_repeat_factor,
        "validation_sample_count": len(val_dataset),
        "best_epoch": best_epoch,
        "best_selection_dice_macro": best_dice,
        "final_best_checkpoint_validation": final_validation,
        "instance_diagnostics": instance_diagnostics,
        "oof_instances": oof_instances,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "split_manifest_sha256": _sha256(split_manifest),
        "config_sha256": _sha256(run_dir / "config.json"),
        "checkpoint_best_sha256": _sha256(best_path),
        "annotation_sha256": annotation_index.annotation_sha256,
        "oof_probability_count": len(probabilities),
        "oof_probability_hashes": probability_hashes,
        "feature_cache": feature_cache,
        "runtime": _runtime_metadata(torch, device),
    }
    _write_json(run_dir / "metrics.json", result)
    _write_json(
        run_dir / "run_state.json",
        {"status": "completed", "run_kind": result["run_kind"]},
    )
    return result


def predict_test(
    checkpoint_path: Path,
    *,
    data_root: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    import torch

    device = _select_device(torch, device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("checkpoint lacks the frozen M2 configuration")
    config = config_from_mapping(checkpoint["config"])
    _seed_everything(torch, config.seed)
    model = build_model(config.with_overrides(encoder_weights=None)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = M2TestDataset(
        data_root,
        image_size=config.image_size,
        normalization_percentiles=(config.percentile_low, config.percentile_high),
        **_feature_dataset_options(config),
    )
    loader = _build_loader(
        torch,
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        seed=config.seed,
    )
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"prediction output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "test_probabilities"
    probabilities: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                logits = model(prepare_model_input(image, config))
            batch_probability = torch.sigmoid(logits).float()[:, 0].cpu().numpy()
            for image_id, probability in zip(
                batch["image_id"], batch_probability, strict=True
            ):
                probabilities[image_id] = np.ascontiguousarray(probability, dtype=np.float32)
    del loader
    gc.collect()
    probability_hashes = _write_probability_directory(probability_dir, probabilities)
    submission = _write_instance_outputs(
        probability_dir,
        dataset.image_ids,
        output_dir / "submission.csv",
        output_dir / "test_instance_manifest.csv",
        config,
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "test_image_count": len(dataset),
        "probability_count": len(probabilities),
        "probability_hashes": probability_hashes,
        "submission": submission,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": _runtime_metadata(torch, device),
    }
    _write_json(output_dir / "prediction_metadata.json", result)
    return result


def evaluate_labeled_split(
    checkpoint_path: Path,
    *,
    data_root: Path,
    split_manifest: Path,
    output_dir: Path,
    target_cache_root: Path,
    split: str,
    device_name: str,
) -> dict[str, Any]:
    """Evaluate one frozen labeled split without changing model or thresholds."""

    import torch

    if split not in {"val", "temporal_holdout"}:
        raise ValueError("labeled evaluation split must be val or temporal_holdout")
    device = _select_device(torch, device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("checkpoint lacks the frozen M2 configuration")
    config = config_from_mapping(checkpoint["config"])
    _seed_everything(torch, config.seed)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = M2SegmentationDataset(
        data_root,
        split_manifest=split_manifest,
        split=split,
        fold=config.fold,
        image_size=config.image_size,
        target_mode=config.target_mode,
        normalization_percentiles=(config.percentile_low, config.percentile_high),
        target_cache_root=target_cache_root,
        **_feature_dataset_options(config),
    )
    dataset.build_target_cache()
    loader = _build_loader(
        torch,
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        seed=config.seed,
    )
    model = build_model(config.with_overrides(encoder_weights=None)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_function = build_bce_dice_loss(config).to(device)
    started = time.perf_counter()
    semantic, probabilities = _validate(
        torch,
        model,
        loader,
        loss_function,
        config,
        device,
        collect_probabilities=True,
        max_steps=None,
    )
    del loader
    gc.collect()
    probability_dir = output_dir / "probabilities"
    probability_hashes = _write_probability_directory(probability_dir, probabilities)
    instance_diagnostics = _write_oof_instance_diagnostics(
        output_dir / "instance_diagnostics.csv",
        probabilities,
        dataset.annotation_index,
        config,
    )
    instance_outputs = _write_instance_outputs(
        probability_dir,
        dataset.image_ids,
        output_dir / "instances.csv",
        output_dir / "instance_manifest.csv",
        config,
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": split,
        "selection_policy": (
            "milestone_only_do_not_tune" if split == "temporal_holdout" else "fold_validation"
        ),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split_manifest_sha256": _sha256(split_manifest),
        "annotation_sha256": dataset.annotation_index.annotation_sha256,
        "semantic_diagnostics": semantic,
        "instance_diagnostics": instance_diagnostics,
        "instance_outputs": instance_outputs,
        "probability_count": len(probabilities),
        "probability_hashes": probability_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": _runtime_metadata(torch, device),
    }
    _write_json(output_dir / "evaluation_metadata.json", result)
    return result


def _add_common_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("cache-targets")
    _add_common_data_arguments(cache_parser)
    cache_parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    cache_parser.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser("train")
    _add_common_data_arguments(train_parser)
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_parser.add_argument("--run-dir", type=Path, default=Path("artifacts/m2/fold0"))
    train_parser.add_argument("--target-cache", type=Path, default=DEFAULT_CACHE)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--skip-cache", action="store_true")
    train_parser.add_argument("--feature-cache", type=Path)
    train_parser.add_argument("--init-checkpoint", type=Path)
    train_parser.add_argument(
        "--train-repeat-factor",
        type=int,
        default=None,
        help="repeat each training sample this many times per epoch",
    )

    predict_parser = subparsers.add_parser("predict-test")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    predict_parser.add_argument("--output", type=Path, default=Path("artifacts/m2/fold0/test"))
    predict_parser.add_argument("--device", default="cuda:0")

    evaluate_parser = subparsers.add_parser("evaluate")
    _add_common_data_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--target-cache", type=Path, default=DEFAULT_CACHE)
    evaluate_parser.add_argument(
        "--split", choices=("val", "temporal_holdout"), default="temporal_holdout"
    )
    evaluate_parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args(argv)
    if args.command == "cache-targets":
        paths = build_target_cache(
            args.data_root,
            split_manifest=args.split_manifest,
            output_root=args.output,
            overwrite=args.overwrite,
        )
        print(json.dumps({"cached_targets": len(paths), "output": str(args.output)}))
        return 0
    if args.command == "train":
        result = train_baseline(
            read_config(args.config),
            data_root=args.data_root,
            split_manifest=args.split_manifest,
            run_dir=args.run_dir,
            target_cache_root=args.target_cache,
            device_name=args.device,
            smoke=args.smoke,
            skip_cache=args.skip_cache,
            feature_cache_dir=args.feature_cache,
            train_repeat_factor=args.train_repeat_factor,
            init_checkpoint=args.init_checkpoint,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "evaluate":
        result = evaluate_labeled_split(
            args.checkpoint,
            data_root=args.data_root,
            split_manifest=args.split_manifest,
            output_dir=args.output,
            target_cache_root=args.target_cache,
            split=args.split,
            device_name=args.device,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    result = predict_test(
        args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output,
        device_name=args.device,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
