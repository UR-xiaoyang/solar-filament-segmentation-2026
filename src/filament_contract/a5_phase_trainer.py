"""A5-4 unified phase trainer for architectural experiments.

This module provides a reusable training framework for all A5-4 phases (WP1-WP4).
Each phase specifies its model architecture, input channels, and loss function,
while training hyperparameters remain frozen from the M2 baseline.

**Frozen training protocol** (from M2 baseline):
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Batch size: 8
- Augmentation: D4 (random 90° rotation + horizontal flip)
- Input: 768×768 resize with reflection padding
- Loss: BCE + soft Dice (1:1 weight, configurable per phase)
- Validation: Every epoch, checkpoint selection by validation Dice
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .a5_pq_contract import compute_pq_metrics, aggregate_pq_metrics


@dataclass
class PhaseTrainingConfig:
    """Training configuration for one A5-4 phase."""

    # Phase identification
    phase_name: str  # e.g., "phase1_fpn", "phase2_contrast"
    fold: int
    output_dir: Path

    # Data
    data_root: Path
    split_manifest_path: Path
    annotation_file: Path

    # Model factory
    model_builder: Callable[[], nn.Module]
    input_channels: int = 1  # Grayscale baseline, Phase 2 uses 2

    # Loss factory (default BCE + Dice, Phase 3 adds clDice)
    loss_builder: Callable[[], nn.Module] | None = None

    # Training (frozen from M2)
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda:0"

    # Smoke test mode
    smoke: bool = False  # If True, run 3 epochs only

    # Random seed for reproducibility
    seed: int = 42


class PhaseTrainer:
    """Unified trainer for A5-4 architectural phases."""

    def __init__(self, config: PhaseTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Build model
        self.model = config.model_builder().to(self.device)

        # Build loss
        if config.loss_builder is not None:
            self.criterion = config.loss_builder().to(self.device)
        else:
            self.criterion = self._default_loss()

        # Build optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.epoch = 0
        self.best_validation_dice = 0.0
        self.best_epoch = 0
        self.history = []

    def _default_loss(self) -> nn.Module:
        """Default BCE + soft Dice loss (M2 baseline)."""
        # Placeholder - will import from archived M2 modules
        # For now, return a simple BCE loss
        return nn.BCEWithLogitsLoss()

    def train(self):
        """Execute full training loop."""
        start_time = time.time()

        print(f"Starting {self.config.phase_name} fold {self.config.fold} training")
        print(f"Output: {self.config.output_dir}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs if not self.config.smoke else 3}")

        # Build dataloaders (placeholder - needs M2 data pipeline)
        train_loader, val_loader = self._build_dataloaders()

        epochs = 3 if self.config.smoke else self.config.epochs

        for epoch in range(1, epochs + 1):
            self.epoch = epoch

            # Train one epoch
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._validate_epoch(val_loader)

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
            self.history.append(epoch_record)

            # Checkpoint selection
            val_dice = val_metrics.get("dice_macro", 0.0)
            if val_dice > self.best_validation_dice:
                self.best_validation_dice = val_dice
                self.best_epoch = epoch
                self._save_checkpoint("best")

            # Log progress
            print(f"Epoch {epoch}/{epochs} - "
                  f"train_loss: {train_metrics['loss']:.4f}, "
                  f"val_dice: {val_dice:.4f}, "
                  f"best: {self.best_validation_dice:.4f} @epoch{self.best_epoch}")

        # Save final checkpoint
        self._save_checkpoint("last")

        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed:.1f}s")

        # Generate OOF probabilities and compute PQ
        oof_metrics = self._generate_oof_and_compute_pq(val_loader)

        # Write final metrics
        self._write_final_metrics(elapsed, oof_metrics)

        return {
            "best_epoch": self.best_epoch,
            "best_validation_dice": self.best_validation_dice,
            "oof_structural_error": oof_metrics.get("structural_error"),
            "elapsed_seconds": elapsed,
        }

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build train and validation dataloaders."""
        # Placeholder - needs M2 data pipeline integration
        # For now, return dummy loaders
        print("WARNING: Using placeholder dataloaders")

        # Create dummy dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                x = torch.randn(self.config.input_channels, 768, 768)
                y = torch.randint(0, 2, (768, 768)).float()
                return x, y

        train_dataset = DummyDataset(100)
        val_dataset = DummyDataset(20)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, val_loader

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets.unsqueeze(1))

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / n_batches}

    def _validate_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_intersection = 0
        total_union = 0
        n_batches = 0

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets.unsqueeze(1))

                # Compute Dice components
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                intersection = (preds * targets.unsqueeze(1)).sum()
                union = preds.sum() + targets.sum()

                total_loss += loss.item()
                total_intersection += intersection.item()
                total_union += union.item()
                n_batches += 1

        dice = 2 * total_intersection / (total_union + 1e-8)

        return {
            "loss": total_loss / n_batches,
            "dice_macro": dice,
        }

    def _generate_oof_and_compute_pq(self, loader: DataLoader) -> dict[str, Any]:
        """Generate OOF probabilities and compute PQ metrics."""
        print("Generating OOF probabilities and computing PQ metrics...")

        self.model.eval()
        per_image_metrics = []

        with torch.no_grad():
            for idx, (images, targets) in enumerate(loader):
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()

                # Compute PQ for each image in batch
                for b in range(probs.shape[0]):
                    prob_map = probs[b, 0]  # (H, W)
                    target_mask = targets[b].numpy()

                    # For dummy data, no ground truth instances
                    # In real implementation, load from COCO annotations
                    gt_masks = []

                    try:
                        metrics = compute_pq_metrics(
                            prob_map,
                            gt_masks,
                            f"dummy_{idx}_{b}",
                        )
                        per_image_metrics.append(metrics)
                    except Exception as e:
                        print(f"Warning: PQ computation failed for image {idx}_{b}: {e}")

        if per_image_metrics:
            aggregate = aggregate_pq_metrics(per_image_metrics)
            return aggregate
        else:
            return {"structural_error": None}

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.config.output_dir / f"checkpoint_{name}.pt"
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_validation_dice": self.best_validation_dice,
            "best_epoch": self.best_epoch,
        }, checkpoint_path)

    def _write_final_metrics(self, elapsed_seconds: float, oof_metrics: dict):
        """Write final training metrics to JSON."""
        metrics = {
            "phase_name": self.config.phase_name,
            "fold": self.config.fold,
            "best_epoch": self.best_epoch,
            "best_validation_dice": self.best_validation_dice,
            "oof_pq_metrics": oof_metrics,
            "elapsed_seconds": elapsed_seconds,
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "input_channels": self.config.input_channels,
                "smoke": self.config.smoke,
            },
        }

        metrics_path = self.config.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Wrote metrics: {metrics_path}")


__all__ = [
    "PhaseTrainingConfig",
    "PhaseTrainer",
]
