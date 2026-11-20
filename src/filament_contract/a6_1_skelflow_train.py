"""
A6-1 SkelFlow Training Pipeline

Training script for SkelFlow model with multi-task learning.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from filament_contract.a6_1_skelflow_model import build_skelflow_model
from filament_contract.a6_1_skelflow_loss import build_skelflow_loss
from filament_contract.a6_1_skelflow_dataset import build_skelflow_dataloader


class SkelFlowTrainer:
    """Trainer for SkelFlow model."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        run_dir: Path,
        log_interval: int = 10,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        self.log_interval = log_interval

        # Setup logging
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir / "logs")

        # Best checkpoint tracking
        self.best_val_loss = float('inf')
        self.epoch = 0

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        metrics = {
            'loss': 0.0,
            'skeleton_loss': 0.0,
            'flow_loss': 0.0,
            'width_loss': 0.0,
            'topology_loss': 0.0,
        }

        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [train]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            targets = {
                'skeleton': batch['skeleton'].to(self.device),
                'flow_field': batch['flow_field'].to(self.device),
                'width_map': batch['width_map'].to(self.device),
            }

            # Forward pass
            predictions = self.model(images)

            # Compute loss
            loss_dict = self.loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()

            # Accumulate metrics
            for key in metrics.keys():
                metrics[key] += loss_dict[key].item()
            num_batches += 1

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': loss_dict['loss'].item(),
                    'skel': loss_dict['skeleton_loss'].item(),
                    'flow': loss_dict['flow_loss'].item(),
                })

        # Average metrics
        for key in metrics.keys():
            metrics[key] /= num_batches

        return metrics

    def validate(self) -> dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        metrics = {
            'loss': 0.0,
            'skeleton_loss': 0.0,
            'flow_loss': 0.0,
            'width_loss': 0.0,
            'topology_loss': 0.0,
        }

        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [val]")
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                targets = {
                    'skeleton': batch['skeleton'].to(self.device),
                    'flow_field': batch['flow_field'].to(self.device),
                    'width_map': batch['width_map'].to(self.device),
                }

                # Forward pass
                predictions = self.model(images)

                # Compute loss
                loss_dict = self.loss_fn(predictions, targets)

                # Accumulate metrics
                for key in metrics.keys():
                    metrics[key] += loss_dict[key].item()
                num_batches += 1

        # Average metrics
        for key in metrics.keys():
            metrics[key] /= num_batches

        return metrics

    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Run directory: {self.run_dir}")

        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch_time)

            # Save checkpoint
            self._save_checkpoint(val_metrics['loss'])

            print(f"\nEpoch {self.epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Val loss: {val_metrics['loss']:.4f}")

        print(f"\nTraining complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")

        self.writer.close()

    def _log_metrics(self, train_metrics: dict, val_metrics: dict, epoch_time: float):
        """Log metrics to tensorboard and file."""
        # Tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.epoch)

        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, self.epoch)

        self.writer.add_scalar("epoch_time", epoch_time, self.epoch)

        if self.scheduler is not None:
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], self.epoch)

        # Write to JSON log
        log_entry = {
            'epoch': self.epoch,
            'train': train_metrics,
            'val': val_metrics,
            'epoch_time': epoch_time,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        log_file = self.run_dir / "training_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _save_checkpoint(self, val_loss: float):
        """Save checkpoint if validation loss improved."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save last checkpoint
        last_path = self.run_dir / "checkpoint_last.pt"
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.run_dir / "checkpoint_best.pt"
            shutil.copy(last_path, best_path)
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")


def load_config(config_path: Path) -> dict:
    """Load training configuration from JSON."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train SkelFlow model")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to config JSON")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to MAGFiLO dataset")
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="Path to SkelFlow GT directory")
    parser.add_argument("--split-manifest", type=Path, required=True,
                        help="Path to split manifest")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (cuda:0, cpu)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test mode (5 epochs, small subset)")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.smoke:
        print("🔥 SMOKE TEST MODE 🔥")
        config['epochs'] = 5

    print(f"\nConfig:")
    print(json.dumps(config, indent=2))

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Build model
    print("\nBuilding model...")
    model = build_skelflow_model(pretrained=config.get('pretrained', True))
    model = model.to(device)

    # Build loss
    loss_fn = build_skelflow_loss(
        weight_skeleton=config.get('weight_skeleton', 0.4),
        weight_flow=config.get('weight_flow', 0.3),
        weight_width=config.get('weight_width', 0.2),
        weight_topology=config.get('weight_topology', 0.1),
    )

    # Build dataloaders
    print("\nBuilding dataloaders...")
    train_loader = build_skelflow_dataloader(
        data_root=args.data_root,
        gt_dir=args.gt_dir,
        split_manifest=args.split_manifest,
        fold=config['fold'],
        role='development',  # Use 'development' role from split manifest
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config.get('num_workers', 4),
        shuffle=True,
    )

    val_loader = build_skelflow_dataloader(
        data_root=args.data_root,
        gt_dir=args.gt_dir,
        split_manifest=args.split_manifest,
        fold=config['fold'],
        role='development',  # Use 'development' role for validation too
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config.get('num_workers', 4),
        shuffle=False,
    )

    # Build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4),
    )

    # Build scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config.get('min_lr', 1e-6),
    )

    # Build trainer
    trainer = SkelFlowTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=args.run_dir,
        log_interval=config.get('log_interval', 10),
    )

    # Save config
    config_save_path = args.run_dir / "config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Train
    trainer.train(num_epochs=config['epochs'])


if __name__ == "__main__":
    main()
