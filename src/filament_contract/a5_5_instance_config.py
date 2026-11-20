"""A5-5 frozen configuration schema for the instance-aware architecture phase.

This module defines the strict schema for A5-5 Mask R-CNN based instance
segmentation. It is intentionally separate from the M2 semantic segmentation
schema (in ``m2_config``) to preserve single-variable experiment discipline:

* A5-4 Control-768 baseline (M2 schema 1, semantic) — frozen comparator
* A5-5 instance-aware (this schema, instance) — new experimental arm

Schema version 1 is the only authorized version until a future schema
migration is explicitly approved. The schema validates:

* Frozen M1 split manifest hash (reused from M2)
* Filament-adapted anchor geometry (long thin structures)
* Frozen M2 training protocol (AdamW, lr=3e-4, batch=8, D4 augmentation)
* Single-class target (background + filament only)
* Deterministic post-processing parameters for evaluation comparability

Why Mask R-CNN with torchvision?
--------------------------------
* No new framework dependencies — uses the standard ``torch`` and
  ``torchvision`` already required in the project's ``train`` extras.
* Anchors can be customized for filament shapes (highly elongated),
  which is the central architectural change.
* Reuses the frozen M1 split manifest and the strict PQ contract so
  that evaluation is directly comparable to the Control-768 baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Tuple


A5_5_SCHEMA_VERSION = 1

# Reused from M2: the frozen M1 split manifest hash. A5-5 must consume
# the same split manifest so that fold assignments are comparable to
# Control-768. Any change to this constant requires an explicit user
# authorization because it alters the comparable evaluation surface.
FROZEN_M1_SPLIT_MANIFEST_SHA256 = (
    "8229294117008071ec704cde609d301c54f0e7b7300c505b3fc9bd83213706ed"
)

# Project-wide metadata fields. Mirrors M2's ``_M12_METADATA_FIELDS``
# pattern: ``config_from_mapping`` accepts these but does NOT type them
# (they are preserved as project bookkeeping only). Adding new entries
# is a non-breaking change.
_METADATA_FIELDS = frozenset(
    {
        "wp_stage",
        "experiment_id",
        "experiment_category",
        "experiment_description",
        "wp_documentation",
    }
)

# Filament-adapted default anchor configuration. Solar filaments are
# long thin dark structures on a bright disk, with typical length:width
# ratios that span roughly 4:1 to 32:1. The default torchvision anchors
# (sizes 32,64,128,256,512; aspect ratios 0.5, 1.0, 2.0) are tuned for
# natural objects and split one long filament into many overlapping
# boxes. A5-5 widens the aspect-ratio menu and shrinks the smallest
# anchor scale because filament widths are usually only a few pixels.
DEFAULT_FILAMENT_ANCHOR_SIZES: Tuple[int, ...] = (16, 32, 64, 128, 256)
DEFAULT_FILAMENT_ASPECT_RATIOS: Tuple[float, ...] = (
    0.125,  # 1:8
    0.25,   # 1:4
    0.5,    # 1:2
    1.0,    # square (control)
    2.0,    # 2:1
    4.0,    # 4:1
    8.0,    # 8:1
)


@dataclass(frozen=True, slots=True)
class A5_5InstanceConfig:
    """All choices that can change an A5-5 training or inference result."""

    # Schema identity
    schema_version: int = A5_5_SCHEMA_VERSION
    experiment_id: str = "a5_5_maskrcnn_filament"

    # Reproducibility
    seed: int = 20260713

    # Data: folds, resolution
    fold: int = 0
    image_size: int = 768  # square backbone input, matching Control-768

    # Architecture: backbone and heads
    backbone: str = "resnet34"
    pretrained_backbone: bool = True
    num_classes: int = 2  # background + filament (single foreground class)

    # Filament-adapted anchor geometry (overridable for ablation)
    anchor_sizes: Tuple[int, ...] = DEFAULT_FILAMENT_ANCHOR_SIZES
    aspect_ratios: Tuple[float, ...] = DEFAULT_FILAMENT_ASPECT_RATIOS
    # Per-FPN-level anchor sizes. When set, overrides the flat
    # ``anchor_sizes`` list and gives each FPN level its own scale,
    # following the standard "anchor size ≈ 4×stride" rule. The
    # default is one size per level; ``None`` falls back to the flat
    # ``anchor_sizes`` replicated across all levels.
    anchor_sizes_per_level: Tuple[Tuple[int, ...], ...] | None = None
    # Region proposal network hyperparameters
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 1000
    rpn_post_nms_top_n_test: int = 500
    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    # ROI head hyperparameters
    box_score_thresh: float = 0.05  # low during inference, then filtered
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 64  # capped to match M2 max_instances_per_image
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    # Mask ROI pooler output size — final mask is 4× this due to two
    # 2× upsamples in MaskRCNNPredictor. Default 14 → 56×56 mask.
    # Bumping to 28 → 112×112 mask gives more cells for thin filaments.
    mask_roi_pooler_output_size: int = 14
    # Switch to FilamentNet (extreme anchors + PixelShuffle mask head).
    # Off by default — the frozen A5-5 baseline uses vanilla torchvision
    # MaskRCNN so that single-variable comparisons to Control-768 are
    # clean. WP-A5-5d enables this to test the specialised architecture.
    use_filament_net: bool = False

    # Training (frozen from M2)
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 4
    augment_d4: bool = True  # random 90° rotation + horizontal flip
    amp: bool = True

    # Post-processing: deterministic instance extraction
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    min_instance_area: int = 256  # matches Control-768 for comparability

    # Evaluation
    evaluation_resolution: Tuple[int, int] = (2048, 2048)
    match_iou_threshold: float = 0.5

    # Frozen split manifest
    expected_manifest_sha256: str = FROZEN_M1_SPLIT_MANIFEST_SHA256

    # Run-mode knobs
    smoke: bool = False
    train_repeat_factor: int = 1

    # ----- Validation -----

    def __post_init__(self) -> None:
        # Validate eagerly so that constructing an invalid config fails
        # loudly. M2 follows the "validate() called explicitly" pattern,
        # but for a new module we prefer the safer modern convention.
        self.validate()

    def validate(self) -> None:
        """Reject any combination that would invalidate comparison with Control-768."""
        if self.schema_version != A5_5_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported A5-5 schema version {self.schema_version}; "
                f"expected {A5_5_SCHEMA_VERSION}"
            )
        if self.fold not in range(5):
            raise ValueError("fold must be in [0, 4]")
        if self.image_size < 256 or self.image_size % 32:
            raise ValueError("image_size must be >= 256 and divisible by 32")
        if self.backbone not in {"resnet34"}:
            raise ValueError(
                f"backbone must be 'resnet34' to match Control-768; got {self.backbone!r}"
            )
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2 (background + filament)")
        if not self.anchor_sizes or any(size <= 0 for size in self.anchor_sizes):
            raise ValueError("anchor_sizes must be positive")
        if not self.aspect_ratios or any(ratio <= 0 for ratio in self.aspect_ratios):
            raise ValueError("aspect_ratios must be positive")
        if self.box_detections_per_img < 1:
            raise ValueError("box_detections_per_img must be positive")
        if self.mask_roi_pooler_output_size < 7:
            raise ValueError("mask_roi_pooler_output_size must be >= 7")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 < self.score_threshold < 1:
            raise ValueError("score_threshold must be in (0, 1)")
        if not 0 < self.mask_threshold < 1:
            raise ValueError("mask_threshold must be in (0, 1)")
        if self.min_instance_area < 1:
            raise ValueError("min_instance_area must be positive")
        if not 0 < self.match_iou_threshold < 1:
            raise ValueError("match_iou_threshold must be in (0, 1)")
        if len(self.evaluation_resolution) != 2:
            raise ValueError("evaluation_resolution must be (height, width)")
        if any(dim <= 0 for dim in self.evaluation_resolution):
            raise ValueError("evaluation_resolution dims must be positive")
        if (
            isinstance(self.train_repeat_factor, bool)
            or not isinstance(self.train_repeat_factor, int)
            or self.train_repeat_factor < 1
        ):
            raise ValueError("train_repeat_factor must be a positive integer")

    # ----- Mutation -----

    def with_overrides(self, **overrides: Any) -> "A5_5InstanceConfig":
        """Return a new config with overrides applied and revalidated."""
        unknown = set(overrides).difference(field.name for field in fields(self))
        if unknown:
            raise ValueError(f"unknown A5-5 configuration fields: {sorted(unknown)}")
        updated = replace(self, **overrides)
        updated.validate()
        return updated

    # ----- Serialization -----

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def config_from_mapping(values: Mapping[str, Any]) -> A5_5InstanceConfig:
    """Build an A5-5 config from a JSON mapping, filling defaults for omitted fields."""
    expected = {field.name for field in fields(A5_5InstanceConfig)}
    allowed = expected | _METADATA_FIELDS
    unknown = set(values).difference(allowed)
    if unknown:
        raise ValueError(f"unknown A5-5 configuration fields: {sorted(unknown)}")
    # Drop metadata fields before hydration.
    hydrated: dict[str, Any] = {
        **A5_5InstanceConfig().to_dict(),
        **{k: v for k, v in values.items() if k in expected},
    }
    # Coerce tuple-typed fields from JSON lists
    for key in ("anchor_sizes", "aspect_ratios", "evaluation_resolution"):
        if key in hydrated and not isinstance(hydrated[key], tuple):
            hydrated[key] = tuple(hydrated[key])
    config = A5_5InstanceConfig(**hydrated)
    config.validate()
    return config


def read_config(path: Path) -> A5_5InstanceConfig:
    """Read an A5-5 config from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("A5-5 configuration must be a JSON object")
    return config_from_mapping(values)


def write_config(path: Path, config: A5_5InstanceConfig) -> Path:
    """Serialize an A5-5 config to JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "A5_5_SCHEMA_VERSION",
    "A5_5InstanceConfig",
    "DEFAULT_FILAMENT_ANCHOR_SIZES",
    "DEFAULT_FILAMENT_ASPECT_RATIOS",
    "FROZEN_M1_SPLIT_MANIFEST_SHA256",
    "config_from_mapping",
    "read_config",
    "write_config",
]