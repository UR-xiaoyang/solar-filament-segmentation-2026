"""Instrumented M2 training with probes T0-T7.

This is a drop-in extension of m2_pipeline.train_baseline() that adds training
probes without changing optimization behavior. It writes probe telemetry to
experiments/<run_name>/training_probes.jsonl and activation summaries per epoch.

Usage:
    from filament_contract.m2_pipeline_instrumented import train_baseline_instrumented
    train_baseline_instrumented(config, data_root, manifest_path, cache_root, output_root)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .m2_config import M2Config
from .m2_pipeline import (
    _append_jsonl,
    _build_loader,
    _d4_augment,
    _feature_cache_metadata,
    _seed_everything,
    _select_device,
    _sha256,
    _validate,
    _write_json,
    FROZEN_M1_SPLIT_MANIFEST_SHA256,
)
from .m2_data import CocoSemanticIndex, M2SegmentationDataset, build_target_cache
from .m2_model import build_bce_dice_loss, build_model, prepare_model_input
from .m2_augmentation import RepeatedTrainingDataset
from .training_probes import ActivationProbe, TrainingEpochProbe


def _train_epoch_instrumented(
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
    probe: TrainingEpochProbe,
    activation_probe: ActivationProbe | None,
) -> dict[str, float]:
    """Train one epoch with probes T0-T6."""
    model.train()
    from collections import defaultdict
    
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
        
        # T5: Gradient probe
        grad_norm = None
        if step % 10 == 0:  # Sample every 10 batches
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += float(param.grad.data.norm(2).item() ** 2)
            grad_norm = total_norm ** 0.5
        
        # T6: Weight update probe
        update_norm = None
        if step % 10 == 0:
            param_norms = []
            for param in model.parameters():
                param_norms.append(float(param.data.norm(2).item()))
            update_norm = sum(param_norms) / len(param_norms) if param_norms else 0.0
        
        scaler.step(optimizer)
        scaler.update()
        
        batch_size = int(image.shape[0])
        sample_count += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["bce"] += float(parts["bce"]) * batch_size
        totals["soft_dice"] += float(parts["soft_dice"]) * batch_size
        
        # T0-T4: Training epoch probe
        if step % 5 == 0:  # Sample every 5 batches
            probe.observe_batch(
                image=image,
                target=target,
                logits=logits,
                loss_parts=parts,
                grad_norm=grad_norm,
                update_norm=update_norm,
            )
    
    if sample_count == 0:
        raise RuntimeError("training loader produced no samples")
    
    return {key: value / sample_count for key, value in totals.items()}


def train_baseline_instrumented(
    config: M2Config,
    data_root: Path,
    manifest_path: Path,
    cache_root: Path,
    output_root: Path,
) -> Path:
    """Train baseline with full probe instrumentation (T0-T7)."""
    import time
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required") from exc
    
    _seed_everything(torch, config.seed)
    device = _select_device(torch, config.device)
    
    # Verify manifest
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != FROZEN_M1_SPLIT_MANIFEST_SHA256:
        raise RuntimeError(
            f"Manifest {manifest_path} has SHA256 {manifest_sha} "
            f"but frozen M1 contract requires {FROZEN_M1_SPLIT_MANIFEST_SHA256}"
        )
    
    # Build target cache
    cache_root.mkdir(parents=True, exist_ok=True)
    target_cache = build_target_cache(
        data_root=data_root,
        manifest_path=manifest_path,
        cache_root=cache_root,
        fold=config.fold,
        role_filter="development",
    )
    
    # Build datasets
    coco_index = CocoSemanticIndex(
        coco_path=data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        image_root=data_root / "train" / "train_images",
        manifest_path=manifest_path,
    )
    
    train_dataset = M2SegmentationDataset(
        coco_index=coco_index,
        target_cache=target_cache,
        filter_mode="train",
        config=config,
    )
    
    val_dataset = M2SegmentationDataset(
        coco_index=coco_index,
        target_cache=target_cache,
        filter_mode="validation",
        config=config,
    )
    
    if config.repeat_training_samples > 1:
        train_dataset = RepeatedTrainingDataset(
            train_dataset, config.repeat_training_samples
        )
    
    train_loader = _build_loader(
        torch,
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        seed=config.seed,
    )
    
    val_loader = _build_loader(
        torch,
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        seed=config.seed,
    )
    
    # Build model
    model = build_model(config)
    model.to(device)
    
    # Initialize probes
    probe_dir = output_root / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    
    # T2: Activation probe (sample key layers)
    activation_probe = ActivationProbe(
        model,
        include_names={
            "encoder.layer1",
            "encoder.layer2",
            "encoder.layer3",
            "encoder.layer4",
            "decoder",
        } if hasattr(model, "encoder") else None
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    loss_function = build_bce_dice_loss(
        bce_weight=config.bce_weight,
        soft_dice_weight=config.soft_dice_weight,
    )
    
    # Training loop
    run_meta = {
        "config": config.__dict__,
        "manifest_sha256": manifest_sha,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "feature_cache": _feature_cache_metadata(train_dataset),
    }
    
    _write_json(output_root / "run_protocol.json", run_meta)
    history_path = output_root / "history.jsonl"
    
    best_val_loss = float("inf")
    best_epoch = -1
    
    for epoch in range(config.epochs):
        epoch_probe = TrainingEpochProbe(output_dir=probe_dir, epoch=epoch)
        
        # Train
        t0 = time.perf_counter()
        train_metrics = _train_epoch_instrumented(
            torch=torch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_function=loss_function,
            config=config,
            device=device,
            max_steps=None,
            probe=epoch_probe,
            activation_probe=activation_probe,
        )
        train_time = time.perf_counter() - t0
        
        # T0-T6: Write training probes
        training_summary = epoch_probe.write_epoch_summary()
        
        # T2: Activation summary
        activation_summary = activation_probe.summarize_and_clear()
        if activation_summary:
            _write_json(
                probe_dir / f"activation_epoch{epoch:03d}.json",
                activation_summary
            )
        
        # Validate
        t0 = time.perf_counter()
        val_metrics, _ = _validate(
            torch=torch,
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            config=config,
            device=device,
            collect_probabilities=False,
            max_steps=None,
        )
        val_time = time.perf_counter() - t0
        
        scheduler.step()
        
        # History
        history_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "train_time_s": train_time,
            "val_time_s": val_time,
            "lr": scheduler.get_last_lr()[0],
            "probe_summary": {
                "probability_separation": training_summary["outputs"]["probability_separation"],
                "gradient_norm": training_summary["optimization"]["gradient_norm"],
            }
        }
        
        _append_jsonl(history_path, history_record)
        
        # Checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config.__dict__,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, output_root / "best_model.pth")
        
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"prob_sep={training_summary['outputs']['probability_separation']:.3f} "
            f"grad_norm={training_summary['optimization']['gradient_norm']:.2f}"
        )
    
    activation_probe.close()
    
    # Final summary
    _write_json(
        output_root / "training_summary.json",
        {"best_epoch": best_epoch, "best_val_loss": best_val_loss}
    )
    
    return output_root / "best_model.pth"
