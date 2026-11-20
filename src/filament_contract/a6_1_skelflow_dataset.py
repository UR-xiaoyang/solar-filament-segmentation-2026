"""
A6-1 SkelFlow Dataset

PyTorch Dataset for loading images and SkelFlow ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SkelFlowDataset(Dataset):
    """
    Dataset for SkelFlow training.

    Loads:
        - Images from MAGFiLO dataset
        - Skeleton, flow field, and width targets from preprocessed .npz files
    """

    def __init__(
        self,
        data_root: Path,
        gt_dir: Path,
        split_manifest: Path,
        fold: int,
        role: str = "train",
        image_size: int = 512,
        normalize: bool = True,
    ):
        """
        Args:
            data_root: Path to MAGFiLO dataset root
            gt_dir: Path to directory containing preprocessed SkelFlow GT (.npz files)
            split_manifest: Path to M1 split_manifest.csv
            fold: Fold number (0-4)
            role: "train" or "val"
            image_size: Target image size for training
            normalize: Whether to normalize images to [0, 1]
        """
        self.data_root = Path(data_root)
        self.gt_dir = Path(gt_dir)
        self.image_size = image_size
        self.normalize = normalize

        # Load split manifest
        self.image_ids = self._load_split(split_manifest, fold, role)

        print(f"SkelFlowDataset: fold={fold}, role={role}, images={len(self.image_ids)}")

    def _load_split(self, manifest_path: Path, fold: int, role: str) -> list[str]:
        """Load image IDs for the specified fold and role."""
        import csv

        image_ids = []
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip temporal_holdout (has empty fold)
                if row["role"] != role:
                    continue

                # For development role, check fold matches
                if role == "development":
                    if row["fold"] and int(row["fold"]) == fold:
                        image_ids.append(row["image_id"])
                # For other roles (train/val), also check fold
                elif role in ["train", "val"]:
                    if row["fold"] and int(row["fold"]) == fold:
                        image_ids.append(row["image_id"])

        # Filter to only images with GT files
        available_ids = []
        for image_id in image_ids:
            gt_path = self.gt_dir / f"{image_id}.jpeg.npz"
            if gt_path.exists():
                available_ids.append(image_id)

        if len(available_ids) < len(image_ids):
            print(f"  Warning: Only {len(available_ids)}/{len(image_ids)} images have GT generated")

        return available_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Load one sample.

        Returns:
            dict with keys:
                - image: [3, H, W] tensor (float32, normalized if requested)
                - skeleton: [1, H, W] binary tensor
                - flow_field: [2, H, W] normalized flow vectors
                - width_map: [1, H, W] width in pixels
                - image_id: str
        """
        image_id = self.image_ids[idx]

        # Load image
        image_path = self.data_root / "train" / "train_images" / f"{image_id}.jpeg"
        image = Image.open(image_path).convert("L")  # Grayscale

        # Resize to training size
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to numpy
        image_np = np.array(image, dtype=np.float32)

        # Normalize to [0, 1]
        if self.normalize:
            image_np = image_np / 255.0

        # Convert grayscale to 3-channel (for pretrained ResNet)
        image_np = np.stack([image_np, image_np, image_np], axis=0)  # [3, H, W]

        # Load ground truth
        gt_path = self.gt_dir / f"{image_id}.jpeg.npz"
        gt_data = np.load(gt_path)

        skeleton = gt_data["skeleton"].astype(np.float32)  # [H, W]
        flow_field = gt_data["flow_field"].astype(np.float32)  # [2, H, W]
        width_map = gt_data["width_map"].astype(np.float32)  # [H, W]

        # Resize GT to training size
        from scipy.ndimage import zoom

        if skeleton.shape != (self.image_size, self.image_size):
            zoom_factor = self.image_size / skeleton.shape[0]
            skeleton = zoom(skeleton, zoom_factor, order=0)  # Nearest for binary
            width_map = zoom(width_map, zoom_factor, order=1) * zoom_factor  # Scale width

            # Resize flow field (preserve direction)
            flow_field_resized = np.zeros((2, self.image_size, self.image_size), dtype=np.float32)
            for i in range(2):
                flow_field_resized[i] = zoom(flow_field[i], zoom_factor, order=1)

            # Re-normalize flow vectors
            flow_magnitude = np.sqrt(flow_field_resized[0]**2 + flow_field_resized[1]**2) + 1e-6
            flow_field_resized[0] /= flow_magnitude
            flow_field_resized[1] /= flow_magnitude

            flow_field = flow_field_resized

        # Add channel dimension to skeleton and width
        skeleton = skeleton[np.newaxis, ...]  # [1, H, W]
        width_map = width_map[np.newaxis, ...]  # [1, H, W]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_np)
        skeleton_tensor = torch.from_numpy(skeleton)
        flow_field_tensor = torch.from_numpy(flow_field)
        width_map_tensor = torch.from_numpy(width_map)

        return {
            "image": image_tensor,
            "skeleton": skeleton_tensor,
            "flow_field": flow_field_tensor,
            "width_map": width_map_tensor,
            "image_id": image_id,
        }


def build_skelflow_dataloader(
    data_root: Path,
    gt_dir: Path,
    split_manifest: Path,
    fold: int,
    role: str,
    batch_size: int,
    image_size: int = 512,
    num_workers: int = 4,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Build DataLoader for SkelFlow training.

    Args:
        data_root: Path to MAGFiLO dataset
        gt_dir: Path to SkelFlow GT directory
        split_manifest: Path to split manifest
        fold: Fold number
        role: "train" or "val"
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    dataset = SkelFlowDataset(
        data_root=data_root,
        gt_dir=gt_dir,
        split_manifest=split_manifest,
        fold=fold,
        role=role,
        image_size=image_size,
        normalize=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=role == "train",  # Drop last incomplete batch in training
    )

    return dataloader


if __name__ == "__main__":
    # Test dataset
    from pathlib import Path

    data_root = Path("data/raw/MAGFiLO_1.0_Kaggle_2026")
    gt_dir = Path("data/processed/skelflow_gt")
    split_manifest = Path("artifacts/m1/split_manifest.csv")

    print("Testing SkelFlowDataset...")

    # Check if GT directory exists
    if not gt_dir.exists():
        print(f"GT directory not found: {gt_dir}")
        print("Run a6_1_generate_skelflow_gt.py first")
        exit(1)

    dataset = SkelFlowDataset(
        data_root=data_root,
        gt_dir=gt_dir,
        split_manifest=split_manifest,
        fold=0,
        role="train",
        image_size=512,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load first sample
    sample = dataset[0]

    print("\nFirst sample:")
    print(f"  image: {sample['image'].shape}, dtype={sample['image'].dtype}")
    print(f"  skeleton: {sample['skeleton'].shape}, dtype={sample['skeleton'].dtype}")
    print(f"  flow_field: {sample['flow_field'].shape}, dtype={sample['flow_field'].dtype}")
    print(f"  width_map: {sample['width_map'].shape}, dtype={sample['width_map'].dtype}")
    print(f"  image_id: {sample['image_id']}")

    # Check value ranges
    print("\nValue ranges:")
    print(f"  image: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  skeleton: [{sample['skeleton'].min():.3f}, {sample['skeleton'].max():.3f}]")
    print(f"  flow_field: [{sample['flow_field'].min():.3f}, {sample['flow_field'].max():.3f}]")
    print(f"  width_map: [{sample['width_map'].min():.3f}, {sample['width_map'].max():.3f}]")

    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = build_skelflow_dataloader(
        data_root=data_root,
        gt_dir=gt_dir,
        split_manifest=split_manifest,
        fold=0,
        role="train",
        batch_size=4,
        num_workers=0,
        shuffle=False,
    )

    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  image: {batch['image'].shape}")
    print(f"  skeleton: {batch['skeleton'].shape}")
    print(f"  flow_field: {batch['flow_field'].shape}")
    print(f"  width_map: {batch['width_map'].shape}")

    print("\nTest passed!")
