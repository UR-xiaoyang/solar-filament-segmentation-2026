"""A5-4 WP0: M2 baseline re-evaluation under frozen PQ contract.

This module measures M2 baseline performance using the A5-4 PQ contract,
establishing the frozen baseline for all subsequent architectural phases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .a5_pq_contract import (
    PQ_CONTRACT_VERSION,
    PQ_THRESHOLD,
    PQ_MIN_AREA,
    compute_pq_metrics,
    aggregate_pq_metrics,
)


def load_split_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load M1 split manifest and return image metadata by image_id."""
    import csv

    images = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            images[image_id] = {
                "fold": int(row["fold"]),
                "role": row["role"],
                "year": row.get("year"),  # May be missing in older manifests
            }
    return images


def load_coco_annotations(annotation_path: Path) -> dict[str, list[dict]]:
    """Load COCO annotations and group by image_id."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build image_id lookup
    image_id_map = {img["id"]: img["file_name"].replace(".jpg", "")
                    for img in coco_data["images"]}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_coco_id = ann["image_id"]
        image_id = image_id_map[image_coco_id]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    return annotations_by_image


def rasterize_coco_segmentation(
    segmentation: list[list[float]],
    height: int,
    width: int
) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask."""
    from pycocotools import mask as mask_util

    # COCO polygon format: [[x1,y1,x2,y2,...]]
    rles = mask_util.frPyObjects(segmentation, height, width)
    rle = mask_util.merge(rles)
    binary_mask = mask_util.decode(rle)
    return binary_mask.astype(np.uint8)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_baseline_pq(
    data_root: Path,
    split_manifest: Path,
    annotation_file: Path,
    m2_oof_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Re-score M2 baseline under PQ contract for all development folds.

    Args:
        data_root: Path to MAGFiLO data root
        split_manifest: Path to M1 split_manifest.csv
        annotation_file: Path to train.json COCO annotations
        m2_oof_dirs: List of M2 OOF probability directories (one per fold)
        output_dir: Output directory for immutable artifact

    Returns:
        dict with baseline PQ metrics and hashes
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading M1 split manifest...")
    images = load_split_manifest(split_manifest)
    manifest_hash = compute_file_sha256(split_manifest)

    print("Loading COCO annotations...")
    annotations = load_coco_annotations(annotation_file)
    annotation_hash = compute_file_sha256(annotation_file)

    print(f"Processing {len(images)} images...")

    # Collect development images only
    development_images = {
        img_id: meta for img_id, meta in images.items()
        if meta["role"] in ("train", "val")
    }

    print(f"Development set: {len(development_images)} images")

    # Collect per-image metrics
    per_image_metrics = []
    missing_oof = []

    for image_id, meta in sorted(development_images.items()):
        fold = meta["fold"]

        # Find OOF probability for this image (from correct fold)
        oof_prob_path = None
        for oof_dir in m2_oof_dirs:
            candidate = oof_dir / f"{image_id}_probability.npy"
            if candidate.exists():
                # Verify this is the held-out fold
                if f"fold{fold}" in str(oof_dir):
                    oof_prob_path = candidate
                    break

        if oof_prob_path is None:
            missing_oof.append(image_id)
            continue

        # Load OOF probability
        probability_map = np.load(oof_prob_path)

        # Get ground truth instances for this image
        image_annotations = annotations.get(image_id, [])
        ground_truth_masks = []

        for ann in image_annotations:
            if "segmentation" in ann and ann["segmentation"]:
                h, w = probability_map.shape
                mask = rasterize_coco_segmentation(ann["segmentation"], h, w)
                ground_truth_masks.append(mask)

        # Compute PQ metrics
        metrics = compute_pq_metrics(probability_map, ground_truth_masks, image_id)
        per_image_metrics.append(metrics)

    if missing_oof:
        print(f"WARNING: Missing OOF for {len(missing_oof)} images")
        print(f"First 10: {missing_oof[:10]}")

    print(f"Computed PQ for {len(per_image_metrics)} images")

    # Aggregate metrics
    aggregate = aggregate_pq_metrics(per_image_metrics)

    # Write per-image CSV
    per_image_csv = output_dir / "per_image_pq.csv"
    with open(per_image_csv, "w", encoding="utf-8") as f:
        f.write("image_id,prediction_count,target_count,matched_count,"
                "false_positive_count,false_negative_count,split_count,merge_count,"
                "structural_error,matched_dice_mean,matched_iou_mean\n")
        for m in per_image_metrics:
            f.write(f"{m.image_id},{m.prediction_count},{m.target_count},"
                   f"{m.matched_count},{m.false_positive_count},{m.false_negative_count},"
                   f"{m.split_count},{m.merge_count},{m.structural_error:.6f},"
                   f"{m.matched_dice_mean if m.matched_dice_mean is not None else ''},"
                   f"{m.matched_iou_mean if m.matched_iou_mean is not None else ''}\n")

    print(f"Wrote per-image metrics: {per_image_csv}")

    # Write aggregate JSON
    aggregate_json = output_dir / "m2_baseline_pq.json"
    with open(aggregate_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"Wrote aggregate metrics: {aggregate_json}")

    # Write input hash manifest
    input_manifest = {
        "split_manifest_path": str(split_manifest),
        "split_manifest_sha256": manifest_hash,
        "annotation_file_path": str(annotation_file),
        "annotation_file_sha256": annotation_hash,
        "m2_oof_directories": [str(d) for d in m2_oof_dirs],
        "n_images_processed": len(per_image_metrics),
        "n_images_missing_oof": len(missing_oof),
    }

    input_manifest_path = output_dir / "input_hash_manifest.json"
    with open(input_manifest_path, "w", encoding="utf-8") as f:
        json.dump(input_manifest, f, indent=2)

    # Write protocol
    protocol = {
        "contract_version": PQ_CONTRACT_VERSION,
        "threshold": PQ_THRESHOLD,
        "min_area": PQ_MIN_AREA,
        "metric_scope": "development_oof_instance_proxy_not_official",
        "baseline_model": "M2_768_unetpp_resnet34",
        "test_ground_truth_used": False,
        "temporal_holdout_used_for_selection": False,
    }

    protocol_path = output_dir / "protocol.json"
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    # Summary
    result = {
        "baseline_structural_error": aggregate["structural_error"],
        "baseline_fp_per_image": aggregate["fp_per_image"],
        "baseline_fn_per_image": aggregate["fn_per_image"],
        "baseline_split_per_image": aggregate["split_per_image"],
        "baseline_merge_per_image": aggregate["merge_per_image"],
        "baseline_matched_dice": aggregate["matched_dice_mean"],
        "n_images": aggregate["n_images"],
        "artifact_path": str(output_dir),
    }

    print("\n" + "="*60)
    print("M2 BASELINE PQ METRICS (Development OOF)")
    print("="*60)
    print(f"Structural Error: {result['baseline_structural_error']:.4f}")
    print(f"FP/image:         {result['baseline_fp_per_image']:.4f}")
    print(f"FN/image:         {result['baseline_fn_per_image']:.4f}")
    print(f"Split/image:      {result['baseline_split_per_image']:.4f}")
    print(f"Merge/image:      {result['baseline_merge_per_image']:.4f}")
    print(f"Matched Dice:     {result['baseline_matched_dice']:.4f}")
    print(f"N images:         {result['n_images']}")
    print("="*60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="A5-4 WP0: Measure M2 baseline under PQ contract"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to MAGFiLO_1.0_Kaggle_2026 data root",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="Path to M1 split_manifest.csv",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        help="Path to train.json (default: data_root/train.json)",
    )
    parser.add_argument(
        "--m2-oof",
        type=Path,
        action="append",
        required=True,
        help="M2 OOF directory (repeat for each fold, e.g., artifacts/m3/exp001-resolution768-fold0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for baseline PQ artifact",
    )

    args = parser.parse_args()

    if args.annotation_file is None:
        args.annotation_file = args.data_root / "train.json"

    validate_baseline_pq(
        args.data_root,
        args.split_manifest,
        args.annotation_file,
        args.m2_oof,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
