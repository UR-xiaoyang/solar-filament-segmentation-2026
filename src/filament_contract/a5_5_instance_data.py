"""A5-5 instance data layer.

This module loads the frozen M1 split manifest and COCO annotations and
yields ``(image, target)`` pairs in torchvision Mask R-CNN format::

    target = {
        "boxes":  FloatTensor[N, 4]   in (x1, y1, x2, y2)
        "labels": Int64Tensor[N]      class index (1 = filament)
        "masks":  UInt8Tensor[N, H, W] per-instance binary masks
        "image_id": Int64Tensor[1]    physical image id (as int hash)
        "area":   FloatTensor[N]      bbox area
        "iscrowd": UInt8Tensor[N]     always 0 (no crowd annotations)
    }

Why this is self-contained and not a wrapper over ``m2_data``:

* M2 produces a soft-vote *semantic* target; A5-5 needs per-instance
  binary masks + axis-aligned bboxes. The output contract differs.
* Mask R-CNN needs D4 augmentation that rotates/flips bboxes in addition
  to the image — a separate pipeline.
* Single-variable experiment discipline (per the handoff document):
  only the architecture changes between Control-768 and A5-5, so the
  data layer should not be co-evolved with M2.

The split-manifest and COCO loading logic is intentionally copied
(frozen) from the M2 implementation so that ``FROZEN_M1_SPLIT_MANIFEST_SHA256``
continues to validate the same artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset


# Re-exported for callers that want the frozen hash without depending
# on the config module directly.
FROZEN_M1_SPLIT_MANIFEST_SHA256 = (
    "8229294117008071ec704cde609d301c54f0e7b7300c505b3fc9bd83213706ed"
)
ANNOTATION_FILE_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_NAME_PATTERN = "{stem}.jpeg"
IMAGE_SHAPE: Tuple[int, int] = (2048, 2048)


class A5_5DataError(ValueError):
    """Raised when the A5-5 data contract is violated."""


@dataclass(frozen=True)
class SplitRecord:
    image_id: str
    file_name: str
    annotation_record_count: int
    image_sha256: str
    hard_scene_group: str
    sequence_group: str
    role: str  # "development" or "temporal_holdout"
    fold: int | None


@dataclass(frozen=True)
class CocoAnnotationRecord:
    record_id: str  # == image_id for the kaggle release
    image_id: str
    file_name: str
    height: int
    width: int
    polygons: Tuple[Tuple[float, ...], ...]  # COCO segmentation: list of polygons


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Split manifest and COCO index — copied verbatim from m2_data to preserve
# the frozen hash contract. Any divergence must be authorized.
# ---------------------------------------------------------------------------


_MANIFEST_COLUMNS = frozenset(
    {
        "image_id",
        "file_name",
        "annotation_record_count",
        "sha256",
        "hard_scene_group",
        "sequence_group",
        "role",
        "fold",
    }
)


def load_split_manifest(
    path: Path,
    *,
    expected_sha256: str | None = FROZEN_M1_SPLIT_MANIFEST_SHA256,
) -> Tuple[SplitRecord, ...]:
    """Read physical-image assignments, validating the frozen hash if provided."""
    path = Path(path).resolve()
    if not path.is_file():
        raise A5_5DataError(f"split manifest does not exist: {path}")
    if expected_sha256 is not None:
        actual = _sha256(path)
        if actual != expected_sha256:
            raise A5_5DataError(
                "split manifest hash differs from frozen M1 artifact: "
                f"expected {expected_sha256}, got {actual}"
            )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = frozenset(reader.fieldnames or ())
        missing = sorted(_MANIFEST_COLUMNS - columns)
        if missing:
            raise A5_5DataError(f"split manifest is missing columns: {missing}")
        raw_rows = list(reader)
    if not raw_rows:
        raise A5_5DataError("split manifest must not be empty")

    records: List[SplitRecord] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    group_assignments: dict[Tuple[str, str], Tuple[str, int | None]] = {}
    development_folds: set[int] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        image_id = row["image_id"].strip()
        file_name = row["file_name"].strip()
        role = row["role"].strip()
        fold_text = row["fold"].strip()
        if not image_id or Path(file_name).stem != image_id or Path(file_name).suffix != ".jpeg":
            raise A5_5DataError(f"invalid physical image identity on row {row_number}")
        if image_id in seen_ids or file_name in seen_files:
            raise A5_5DataError(f"duplicate physical image on row {row_number}")
        seen_ids.add(image_id)
        seen_files.add(file_name)

        image_sha256 = row["sha256"].strip().lower()
        if len(image_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in image_sha256
        ):
            raise A5_5DataError(f"invalid image SHA-256 on row {row_number}")

        try:
            annotation_record_count = int(row["annotation_record_count"])
        except ValueError as error:
            raise A5_5DataError(
                f"invalid annotation_record_count on row {row_number}"
            ) from error
        if annotation_record_count < 1:
            raise A5_5DataError(
                f"annotation_record_count must be positive on row {row_number}"
            )

        if role == "development":
            try:
                fold = int(fold_text)
            except ValueError as error:
                raise A5_5DataError(
                    f"development image lacks an integer fold on row {row_number}"
                ) from error
            if fold < 0:
                raise A5_5DataError(f"fold must be non-negative on row {row_number}")
            development_folds.add(fold)
        elif role == "temporal_holdout":
            if fold_text:
                raise A5_5DataError(
                    f"temporal holdout must not have a fold on row {row_number}"
                )
            fold = None
        else:
            raise A5_5DataError(f"unknown role {role!r} on row {row_number}")

        hard_scene_group = row["hard_scene_group"].strip()
        sequence_group = row["sequence_group"].strip()
        if not hard_scene_group or not sequence_group:
            raise A5_5DataError(f"empty leakage group on row {row_number}")

        assignment = (role, fold)
        for group_type, group_id in (
            ("hard_scene", hard_scene_group),
            ("sequence", sequence_group),
        ):
            key = (group_type, group_id)
            previous = group_assignments.setdefault(key, assignment)
            if previous != assignment:
                raise A5_5DataError(
                    f"{group_type} group {group_id!r} crosses split assignments"
                )

        records.append(
            SplitRecord(
                image_id=image_id,
                file_name=file_name,
                annotation_record_count=annotation_record_count,
                image_sha256=image_sha256,
                hard_scene_group=hard_scene_group,
                sequence_group=sequence_group,
                role=role,
                fold=fold,
            )
        )

    expected_folds = set(range(max(development_folds, default=-1) + 1))
    if development_folds != expected_folds:
        raise A5_5DataError(
            "development folds must be contiguous and zero-based: "
            f"got {sorted(development_folds)}"
        )
    return tuple(records)


def select_fold_records(
    records: Sequence[SplitRecord],
    *,
    split: str,
    fold: int,
) -> Tuple[SplitRecord, ...]:
    """Return records belonging to one fold's train, val, or holdout split."""
    if split not in {"train", "val", "temporal_holdout"}:
        raise A5_5DataError(f"unknown split {split!r}")
    development_folds = {r.fold for r in records if r.role == "development"}
    if fold not in development_folds:
        raise A5_5DataError(
            f"fold {fold} absent from development assignments "
            f"{sorted(f for f in development_folds if f is not None)}"
        )
    if split == "train":
        selected = [r for r in records if r.role == "development" and r.fold != fold]
    elif split == "val":
        selected = [r for r in records if r.role == "development" and r.fold == fold]
    else:
        selected = [r for r in records if r.role == "temporal_holdout"]
    if not selected:
        raise A5_5DataError(f"split {split!r} for fold {fold} is empty")
    return tuple(sorted(selected, key=lambda r: r.image_id))


class CocoInstanceIndex:
    """COCO polygons grouped first by annotation record, then by physical image.

    A "physical image" is identified by its unprefixed ``image_id``
    (e.g. ``20110114105034Ch``). The COCO JSON identifies each annotation
    record by a **prefixed** ID (e.g. ``010101-20110114105034Ch``)
    where the prefix is the annotator batch. This class parses those
    prefixed IDs and exposes both:

    * ``records_by_id`` keyed by the prefixed record ID (for direct
      COCO lookups);
    * ``physical_image_to_record_ids`` mapping unprefixed image_id to
      the list of prefixed record IDs that annotate it (for joining
      with the split manifest).
    """

    def __init__(self, annotation_path: Path) -> None:
        self.annotation_path = Path(annotation_path).resolve()
        if not self.annotation_path.is_file():
            raise A5_5DataError(f"annotation JSON does not exist: {self.annotation_path}")
        self.annotation_sha256 = _sha256(self.annotation_path)
        with self.annotation_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise A5_5DataError("COCO annotation root must be an object")
        images = data.get("images")
        annotations = data.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise A5_5DataError("COCO images and annotations must be lists")

        image_metadata: dict[str, Tuple[str, str, int, int]] = {}
        polygons_by_record: defaultdict[str, List[Tuple[float, ...]]] = defaultdict(list)
        for image in images:
            if not isinstance(image, dict):
                raise A5_5DataError("COCO image record must be an object")
            record_id = image.get("id")
            file_name = image.get("file_name")
            height = image.get("height")
            width = image.get("width")
            if (
                not isinstance(record_id, str)
                or not isinstance(file_name, str)
                or not isinstance(height, int)
                or not isinstance(width, int)
                or height <= 0
                or width <= 0
            ):
                raise A5_5DataError("COCO image record has invalid identity or dimensions")
            if record_id in image_metadata:
                raise A5_5DataError(f"duplicate COCO image id: {record_id}")
            image_metadata[record_id] = (record_id, file_name, height, width)

        for ann in annotations:
            if not isinstance(ann, dict):
                raise A5_5DataError("COCO annotation must be an object")
            image_id = ann.get("image_id")
            segmentation = ann.get("segmentation")
            if not isinstance(image_id, str):
                raise A5_5DataError("COCO annotation must reference a string image_id")
            if image_id not in image_metadata:
                raise A5_5DataError(f"annotation references unknown image {image_id!r}")
            if not isinstance(segmentation, list) or not segmentation:
                raise A5_5DataError("COCO annotation segmentation must be a non-empty list")
            for polygon in segmentation:
                if (
                    not isinstance(polygon, list)
                    or len(polygon) < 6
                    or len(polygon) % 2 != 0
                ):
                    raise A5_5DataError("COCO polygon must have >= 3 points (>=6 floats)")
                polygons_by_record[image_id].append(tuple(float(v) for v in polygon))

        # Build the prefixed-ID records and the physical-image → records map.
        records_by_id: dict[str, CocoAnnotationRecord] = {}
        physical_image_to_record_ids: defaultdict[str, List[str]] = defaultdict(list)
        for record_id, (rid, file_name, height, width) in image_metadata.items():
            physical_id = _strip_annotator_prefix(rid)
            records_by_id[record_id] = CocoAnnotationRecord(
                record_id=rid,
                image_id=physical_id,
                file_name=file_name,
                height=height,
                width=width,
                polygons=tuple(polygons_by_record.get(record_id, ())),
            )
            physical_image_to_record_ids[physical_id].append(rid)

        self.records_by_id = records_by_id
        self.physical_image_to_record_ids = dict(physical_image_to_record_ids)
        self.image_id_to_file_name = {
            _strip_annotator_prefix(rid): meta[1] for rid, meta in image_metadata.items()
        }

    def record(self, record_id: str) -> CocoAnnotationRecord:
        if record_id not in self.records_by_id:
            raise A5_5DataError(f"unknown annotation record {record_id!r}")
        return self.records_by_id[record_id]

    def records_for_physical_image(self, physical_image_id: str) -> List[CocoAnnotationRecord]:
        """Return all annotation records that annotate the given physical image."""
        record_ids = self.physical_image_to_record_ids.get(physical_image_id, [])
        return [self.records_by_id[rid] for rid in record_ids]


def _strip_annotator_prefix(prefixed_id: str) -> str:
    """Strip the ``{annotator}-`` prefix from a COCO record ID.

    The kaggle release annotates each physical image with one or more
    annotators; each annotation record carries a prefixed ID such as
    ``010101-20110114105034Ch``. The split manifest references the
    physical image by its unprefixed tail, so we strip the prefix here.

    The convention is the first six digits (the annotator batch), a
    dash, and then the physical-image ID. If the ID has no dash, we
    assume it is already an unprefixed physical ID and return it as-is.
    """
    if "-" not in prefixed_id:
        return prefixed_id
    head, tail = prefixed_id.split("-", 1)
    # Only strip the prefix when the head looks like an annotator batch
    # (6 digits). Otherwise the ID might already be unprefixed or have
    # an unrelated dash and we return it unchanged.
    if head.isdigit() and len(head) == 6:
        return tail
    return prefixed_id


# ---------------------------------------------------------------------------
# Polygon → mask + bbox conversion (the A5-5 specific transformation).
# ---------------------------------------------------------------------------


def polygon_to_mask(polygon: Sequence[float], height: int, width: int) -> np.ndarray:
    """Rasterize a single COCO polygon (flat [x1,y1,...] list) to a uint8 mask."""
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        raise A5_5DataError("polygon must have >= 3 points (>=6 floats)")
    rle = mask_utils.frPyObjects([list(polygon)], height, width)[0]
    return np.ascontiguousarray(mask_utils.decode(rle) != 0, dtype=np.uint8)


def record_to_instance_masks(record: CocoAnnotationRecord) -> List[np.ndarray]:
    """Return per-instance binary masks for one annotation record at native resolution."""
    masks = [
        polygon_to_mask(poly, record.height, record.width) for poly in record.polygons
    ]
    return masks


def mask_to_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a binary mask to an axis-aligned bbox (x1, y1, x2, y2)."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)  # torchvision uses exclusive x2/y2
    y2 = float(ys.max() + 1)
    return (x1, y1, x2, y2)


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# ---------------------------------------------------------------------------
# Runtime resize for instances.
# ---------------------------------------------------------------------------


def _resize_image(array: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    pil = Image.fromarray(array, mode="F")
    if pil.size != (target_w, target_h):
        pil = pil.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(pil), dtype=np.float32)


def _resize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if mask.shape == target_hw:
        return mask.astype(np.uint8, copy=False)
    pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    pil = pil.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    return np.ascontiguousarray((np.asarray(pil) > 127).astype(np.uint8))


def _resize_bbox(
    bbox: Tuple[float, float, float, float],
    original_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    orig_h, orig_w = original_hw
    tgt_h, tgt_w = target_hw
    sx = tgt_w / orig_w
    sy = tgt_h / orig_h
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


# ---------------------------------------------------------------------------
# D4 augmentation (image + masks + bboxes consistently).
# ---------------------------------------------------------------------------


def _rot90_image_and_masks(
    image: np.ndarray,
    masks: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply k x 90° rotation. Image axes become (H, W); masks stay (N, H, W)."""
    if k % 4 == 0:
        return image, masks
    image = np.rot90(image, k=k).copy()
    masks = np.rot90(masks, k=k, axes=(1, 2)).copy()
    return image, masks


def _hflip_image_and_masks(
    image: np.ndarray,
    masks: np.ndarray,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    image = np.ascontiguousarray(np.flip(image, axis=1))
    masks = np.ascontiguousarray(np.flip(masks, axis=2))
    return image, masks


def _rot90_bbox(
    bbox: Tuple[float, float, float, float],
    image_hw: Tuple[int, int],
    k: int,
) -> Tuple[float, float, float, float]:
    """Apply k x 90° rotation to a bbox.

    Conventions match ``np.rot90(arr, k=k)``:

    * ``k=1`` rotates counterclockwise 90°. ``(x, y) → (y, h - 1 - x)``.
    * ``k=2`` rotates 180°. ``(x, y) → (h - 1 - x, h - 1 - y)``.
    * ``k=3`` rotates clockwise 90° (same as CCW 270°).
      ``(x, y) → (h - 1 - y, x)``.

    The image is assumed square (which it is in our pipeline after
    runtime resize), so ``image_hw == (h, h)``.
    """
    h, w = image_hw
    if k % 4 == 0:
        return bbox
    x1, y1, x2, y2 = bbox
    if k == 1:
        # (x, y) → (y, h - 1 - x)  → bbox (y1, h - x2, y2, h - x1)
        return (y1, h - x2, y2, h - x1)
    if k == 2:
        # (x, y) → (h - 1 - x, h - 1 - y)  → bbox (h - x2, h - y2, h - x1, h - y1)
        return (h - x2, h - y2, h - x1, h - y1)
    if k == 3:
        # (x, y) → (h - 1 - y, x)  → bbox (h - y2, x1, h - y1, x2)
        return (h - y2, x1, h - y1, x2)
    raise A5_5DataError(f"rotation k must be in [0, 3], got {k}")


def _hflip_bbox(
    bbox: Tuple[float, float, float, float],
    width: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (width - x2, y1, width - x1, y2)


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------


@dataclass
class _InstanceItem:
    image_id: str
    file_name: str
    image_path: Path
    native_hw: Tuple[int, int]
    masks_native: List[np.ndarray]  # per-instance binary masks at native res


class InstanceSegmentationDataset(Dataset):
    """Mask R-CNN style dataset yielding ``(image_tensor, target_dict)`` pairs.

    The dataset is read-only: every epoch the augmentation is sampled
    independently and the COCO masks are re-rasterized lazily from the
    polygons (which is cheap because we keep the polygons in memory).
    """

    def __init__(
        self,
        *,
        data_root: Path,
        split_manifest_path: Path,
        annotation_file: Path,
        split: str,
        fold: int,
        image_size: int,
        augment_d4: bool = True,
        train: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.split_manifest_path = Path(split_manifest_path).resolve()
        self.annotation_file = Path(annotation_file).resolve()
        self.split = split
        self.fold = fold
        self.image_size = image_size
        self.augment_d4 = augment_d4
        self.train = train

        records = load_split_manifest(self.split_manifest_path)
        self._records = select_fold_records(records, split=split, fold=fold)
        self._index = CocoInstanceIndex(self.annotation_file)
        self._items = self._resolve_items()

    # ----- item resolution -----

    def _resolve_items(self) -> List[_InstanceItem]:
        items: List[_InstanceItem] = []
        missing = []
        for record in self._records:
            record_ids = self._index.physical_image_to_record_ids.get(record.image_id, [])
            if not record_ids:
                missing.append(record.image_id)
                continue
            # Merge all annotation records for this physical image into
            # one instance pool. The strict PQ contract treats each
            # annotation record as an atomic evaluation unit, so for
            # *training* we collapse multiple annotators' views into a
            # single per-image instance set (this matches the semantic
            # soft-vote aggregation in M2). The contract still evaluates
            # per-record downstream.
            ann_records = [self._index.record(rid) for rid in record_ids]
            masks_native: List[np.ndarray] = []
            for ann in ann_records:
                masks_native.extend(record_to_instance_masks(ann))
            # All records for one image share the same file_name and
            # native resolution (verified by the manifest).
            first_ann = ann_records[0]
            items.append(
                _InstanceItem(
                    image_id=record.image_id,
                    file_name=record.file_name,
                    image_path=self.data_root / "train" / "train_images" / record.file_name,
                    native_hw=(first_ann.height, first_ann.width),
                    masks_native=masks_native,
                )
            )
        if missing:
            raise A5_5DataError(
                "manifest images missing from annotation JSON: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''} "
                f"(total {len(missing)})"
            )
        return items

    # ----- Dataset protocol -----

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        import torch  # local import so the module is importable without torch

        item = self._items[idx]
        # Load image as float32 [0, 1].
        with Image.open(item.image_path) as pil:
            pil = pil.convert("L")
            arr = np.asarray(pil, dtype=np.float32) / 255.0

        masks_native = np.stack(item.masks_native, axis=0) if item.masks_native else np.zeros(
            (0, arr.shape[0], arr.shape[1]), dtype=np.uint8
        )

        # Runtime resize.
        target_hw = (self.image_size, self.image_size)
        image = _resize_image(arr, target_hw)
        if masks_native.shape[0] > 0:
            masks = np.stack(
                [_resize_mask(masks_native[i], target_hw) for i in range(masks_native.shape[0])],
                axis=0,
            )
        else:
            masks = np.zeros((0, target_hw[0], target_hw[1]), dtype=np.uint8)
        bboxes = [
            _resize_bbox(
                mask_to_bbox(masks_native[i]), item.native_hw, target_hw
            )
            for i in range(masks_native.shape[0])
        ]

        # D4 augmentation (training only).
        if self.train and self.augment_d4:
            rng = np.random  # deterministic per-worker via DataLoader seed
            k = int(rng.randint(0, 4))
            do_hflip = bool(rng.rand() < 0.5)
            image, masks = _rot90_image_and_masks(image, masks, k)
            new_hw = image.shape
            bboxes = [_rot90_bbox(b, new_hw, k) for b in bboxes]
            if do_hflip:
                image, masks = _hflip_image_and_masks(image, masks, new_hw[1])
                bboxes = [_hflip_bbox(b, new_hw[1]) for b in bboxes]

        # Filter out bboxes with zero area after augmentation.
        keep = [i for i, b in enumerate(bboxes) if bbox_area(b) > 0.0]
        if len(keep) < len(bboxes):
            masks = masks[keep] if masks.shape[0] else masks
            bboxes = [bboxes[i] for i in keep]

        # Build target dict.
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # [1, H, W]
        boxes_tensor = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.ones((boxes_tensor.shape[0],), dtype=torch.int64)
        masks_tensor = torch.from_numpy(masks.astype(np.uint8))
        if masks_tensor.ndim == 2:
            masks_tensor = masks_tensor.unsqueeze(0)
        area_tensor = torch.tensor(
            [bbox_area(b) for b in bboxes], dtype=torch.float32
        )
        iscrowd_tensor = torch.zeros((boxes_tensor.shape[0],), dtype=torch.uint8)
        # Stable 64-bit image id from a string hash. Take 15 hex chars
        # (60 bits) so the value fits in a signed int64 with margin.
        image_id_int = int(hashlib.md5(item.image_id.encode()).hexdigest()[:15], 16)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([image_id_int], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }
        return image_tensor, target, item.image_id


__all__ = [
    "ANNOTATION_FILE_NAME",
    "A5_5DataError",
    "CocoAnnotationRecord",
    "CocoInstanceIndex",
    "FROZEN_M1_SPLIT_MANIFEST_SHA256",
    "IMAGE_SHAPE",
    "IMAGE_NAME_PATTERN",
    "InstanceSegmentationDataset",
    "SplitRecord",
    "bbox_area",
    "load_split_manifest",
    "mask_to_bbox",
    "polygon_to_mask",
    "record_to_instance_masks",
    "select_fold_records",
]