"""Leakage-safe data loading for the M2 semantic segmentation baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from .contract import IMAGE_SHAPE
from .dataset import ANNOTATION_FILE, IMAGE_NAME_PATTERN, DatasetContractError
from .m10_features import LocalContrastFeatureConfig, build_feature_stack
from .m12_data_adapter import build_augmentation_from_config, apply_m12_augmentation


def runtime_image_array(
    image: np.ndarray,
    *,
    image_size: tuple[int, int] | None,
) -> np.ndarray:
    """Apply the image-only resize step shared by runtime features and M10 WP0."""

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise M2DataError("runtime image must be a two-dimensional array")
    if image_size is not None and array.shape != image_size:
        array = _resize(array, image_size, nearest=False)
    return np.ascontiguousarray(array, dtype=np.float32)


FROZEN_M1_SPLIT_MANIFEST_SHA256 = (
    "8229294117008071ec704cde609d301c54f0e7b7300c505b3fc9bd83213706ed"
)
FROZEN_M0_IMAGE_MANIFEST_SHA256 = (
    "0093ffbd5a342f213a33a8b69d6c535835a0a12cd8b62d27ac0a367b6aac0189"
)
FROZEN_ANNOTATION_SHA256 = (
    "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
)

TargetMode = Literal["soft_vote", "majority", "union"]
DatasetSplit = Literal["train", "val", "temporal_holdout"]
TARGET_MODES = frozenset({"soft_vote", "majority", "union"})
DATASET_SPLITS = frozenset({"train", "val", "temporal_holdout"})

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


class M2DataError(DatasetContractError):
    """Raised when M2 data would violate the frozen split or target contract."""


@dataclass(frozen=True)
class SplitRecord:
    image_id: str
    file_name: str
    annotation_record_count: int
    image_sha256: str
    hard_scene_group: str
    sequence_group: str
    role: str
    fold: int | None


@dataclass(frozen=True)
class CocoAnnotationRecord:
    record_id: str
    image_id: str
    file_name: str
    height: int
    width: int
    polygons: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class SampleWindow:
    record: SplitRecord
    top: int
    left: int
    height: int
    width: int


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[SplitRecord, ...]:
    """Read and validate physical-image assignments from the frozen M1 manifest."""

    path = Path(path).resolve()
    if not path.is_file():
        raise M2DataError(f"split manifest does not exist: {path}")
    if expected_sha256 is not None:
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise M2DataError(
                "split manifest hash differs from the frozen M1 artifact: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = frozenset(reader.fieldnames or ())
        missing = sorted(_MANIFEST_COLUMNS - columns)
        if missing:
            raise M2DataError(f"split manifest is missing columns: {missing}")
        raw_rows = list(reader)
    if not raw_rows:
        raise M2DataError("split manifest must not be empty")

    records: list[SplitRecord] = []
    image_ids: set[str] = set()
    file_names: set[str] = set()
    group_assignments: dict[tuple[str, str], tuple[str, int | None]] = {}
    development_folds: set[int] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        image_id = row["image_id"].strip()
        file_name = row["file_name"].strip()
        role = row["role"].strip()
        fold_text = row["fold"].strip()
        if not image_id or Path(file_name).stem != image_id or Path(file_name).suffix != ".jpeg":
            raise M2DataError(f"invalid physical image identity on manifest row {row_number}")
        if image_id in image_ids or file_name in file_names:
            raise M2DataError(f"duplicate physical image on manifest row {row_number}")
        image_ids.add(image_id)
        file_names.add(file_name)
        image_sha256 = row["sha256"].strip().lower()
        if len(image_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in image_sha256
        ):
            raise M2DataError(f"invalid image SHA-256 on manifest row {row_number}")

        try:
            annotation_record_count = int(row["annotation_record_count"])
        except ValueError as error:
            raise M2DataError(
                f"invalid annotation_record_count on manifest row {row_number}"
            ) from error
        if annotation_record_count < 1:
            raise M2DataError(
                f"annotation_record_count must be positive on manifest row {row_number}"
            )

        if role == "development":
            try:
                fold = int(fold_text)
            except ValueError as error:
                raise M2DataError(
                    f"development image lacks an integer fold on row {row_number}"
                ) from error
            if fold < 0:
                raise M2DataError(f"fold must be non-negative on manifest row {row_number}")
            development_folds.add(fold)
        elif role == "temporal_holdout":
            if fold_text:
                raise M2DataError(
                    f"temporal holdout must not have a fold on row {row_number}"
                )
            fold = None
        else:
            raise M2DataError(f"unknown role {role!r} on manifest row {row_number}")

        hard_scene_group = row["hard_scene_group"].strip()
        sequence_group = row["sequence_group"].strip()
        if not hard_scene_group or not sequence_group:
            raise M2DataError(f"empty leakage group on manifest row {row_number}")
        assignment = (role, fold)
        for group_type, group_id in (
            ("hard_scene", hard_scene_group),
            ("sequence", sequence_group),
        ):
            key = (group_type, group_id)
            previous = group_assignments.setdefault(key, assignment)
            if previous != assignment:
                raise M2DataError(
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
        raise M2DataError(
            "development folds must be contiguous and zero-based: "
            f"got {sorted(development_folds)}"
        )
    return tuple(records)


def select_fold_records(
    records: Sequence[SplitRecord],
    *,
    split: DatasetSplit,
    fold: int,
) -> tuple[SplitRecord, ...]:
    """Select physical images for one fold without consulting COCO annotation rows."""

    if split not in DATASET_SPLITS:
        raise M2DataError(f"unknown dataset split: {split!r}")
    development_folds = {record.fold for record in records if record.role == "development"}
    if fold not in development_folds:
        raise M2DataError(
            f"fold {fold} is absent from development assignments "
            f"{sorted(value for value in development_folds if value is not None)}"
        )

    if split == "train":
        selected = [
            record
            for record in records
            if record.role == "development" and record.fold != fold
        ]
    elif split == "val":
        selected = [
            record
            for record in records
            if record.role == "development" and record.fold == fold
        ]
    else:
        selected = [record for record in records if record.role == "temporal_holdout"]
    if not selected:
        raise M2DataError(f"split {split!r} for fold {fold} contains no physical images")
    return tuple(sorted(selected, key=lambda record: record.image_id))


class CocoSemanticIndex:
    """COCO polygons grouped first by annotation record, then physical image."""

    def __init__(self, annotation_path: Path) -> None:
        self.annotation_path = Path(annotation_path).resolve()
        if not self.annotation_path.is_file():
            raise M2DataError(f"annotation JSON does not exist: {self.annotation_path}")
        self.annotation_sha256 = _sha256(self.annotation_path)
        with self.annotation_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise M2DataError("COCO annotation root must be an object")
        images = data.get("images")
        annotations = data.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise M2DataError("COCO images and annotations must be lists")

        image_metadata: dict[str, tuple[str, str, int, int]] = {}
        record_ids_by_image: defaultdict[str, list[str]] = defaultdict(list)
        for image in images:
            if not isinstance(image, dict):
                raise M2DataError("COCO image record must be an object")
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
                raise M2DataError("COCO image record has invalid identity or dimensions")
            if record_id in image_metadata:
                raise M2DataError(f"duplicate COCO image record ID: {record_id}")
            image_id = Path(file_name).stem
            if Path(file_name).suffix != ".jpeg" or not record_id.endswith(f"-{image_id}"):
                raise M2DataError(f"invalid COCO physical image mapping: {record_id}")
            image_metadata[record_id] = (image_id, file_name, height, width)
            record_ids_by_image[image_id].append(record_id)

        polygons_by_record: defaultdict[str, list[tuple[float, ...]]] = defaultdict(list)
        annotation_ids: set[str] = set()
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise M2DataError("COCO annotation must be an object")
            annotation_id = annotation.get("id")
            record_id = annotation.get("image_id")
            segmentation = annotation.get("segmentation")
            if not isinstance(annotation_id, str) or annotation_id in annotation_ids:
                raise M2DataError("COCO annotation IDs must be unique strings")
            annotation_ids.add(annotation_id)
            if record_id not in image_metadata:
                raise M2DataError(
                    f"annotation {annotation_id} references an unknown image record"
                )
            if not isinstance(segmentation, list) or not segmentation:
                raise M2DataError(f"annotation {annotation_id} must contain polygons")
            for polygon in segmentation:
                if (
                    not isinstance(polygon, list)
                    or len(polygon) < 6
                    or len(polygon) % 2
                    or any(not isinstance(value, (int, float)) for value in polygon)
                ):
                    raise M2DataError(f"annotation {annotation_id} has a malformed polygon")
                polygons_by_record[str(record_id)].append(
                    tuple(float(value) for value in polygon)
                )

        missing_annotations = sorted(set(image_metadata) - set(polygons_by_record))
        if missing_annotations:
            raise M2DataError(
                "COCO image records without annotations are unsupported by the frozen "
                f"dataset: {missing_annotations[:3]}"
            )

        records_by_id: dict[str, CocoAnnotationRecord] = {}
        dimensions_by_image: dict[str, tuple[int, int]] = {}
        file_name_by_image: dict[str, str] = {}
        for record_id, (image_id, file_name, height, width) in image_metadata.items():
            previous_dimensions = dimensions_by_image.setdefault(image_id, (height, width))
            previous_file_name = file_name_by_image.setdefault(image_id, file_name)
            if previous_dimensions != (height, width) or previous_file_name != file_name:
                raise M2DataError(
                    f"annotation versions disagree on physical image {image_id}"
                )
            records_by_id[record_id] = CocoAnnotationRecord(
                record_id=record_id,
                image_id=image_id,
                file_name=file_name,
                height=height,
                width=width,
                polygons=tuple(polygons_by_record[record_id]),
            )

        self.records_by_id = records_by_id
        self.record_ids_by_image = {
            image_id: tuple(sorted(record_ids))
            for image_id, record_ids in record_ids_by_image.items()
        }
        self.dimensions_by_image = dimensions_by_image
        self.file_name_by_image = file_name_by_image

    @property
    def image_ids(self) -> frozenset[str]:
        return frozenset(self.record_ids_by_image)

    def validate_manifest(self, records: Sequence[SplitRecord]) -> None:
        manifest_ids = {record.image_id for record in records}
        if manifest_ids != self.image_ids:
            missing_json = sorted(manifest_ids - self.image_ids)
            missing_manifest = sorted(self.image_ids - manifest_ids)
            raise M2DataError(
                "manifest and COCO physical image IDs differ: "
                f"missing_json={missing_json[:3]}, missing_manifest={missing_manifest[:3]}"
            )
        for record in records:
            annotation_record_ids = self.record_ids_by_image[record.image_id]
            if len(annotation_record_ids) != record.annotation_record_count:
                raise M2DataError(
                    f"annotation record count mismatch for {record.image_id}: "
                    f"manifest={record.annotation_record_count}, "
                    f"json={len(annotation_record_ids)}"
                )
            if self.file_name_by_image[record.image_id] != record.file_name:
                raise M2DataError(f"file name mismatch for {record.image_id}")

    def rasterize_record(self, record_id: str) -> np.ndarray:
        """Rasterize one annotator's instances as an idempotent semantic union."""

        try:
            record = self.records_by_id[record_id]
        except KeyError as error:
            raise M2DataError(f"unknown annotation record: {record_id}") from error
        polygons = [list(polygon) for polygon in record.polygons]
        try:
            rles = mask_utils.frPyObjects(polygons, record.height, record.width)
            merged = mask_utils.merge(rles)
            mask = mask_utils.decode(merged)
        except (TypeError, ValueError, IndexError) as error:
            raise M2DataError(f"failed to rasterize annotation record {record_id}") from error
        mask = np.asarray(mask)
        if mask.shape != (record.height, record.width):
            raise M2DataError(f"unexpected rasterized shape for {record_id}: {mask.shape}")
        return np.ascontiguousarray(mask != 0)

    def vote_counts(self, image_id: str) -> tuple[np.ndarray, int]:
        """Return per-pixel annotator votes and the equal-weight denominator."""

        try:
            record_ids = self.record_ids_by_image[image_id]
            height, width = self.dimensions_by_image[image_id]
        except KeyError as error:
            raise M2DataError(f"unknown physical image: {image_id}") from error
        if len(record_ids) > np.iinfo(np.uint8).max:
            raise M2DataError(f"too many annotation records for uint8 votes: {image_id}")
        votes = np.zeros((height, width), dtype=np.uint8)
        for record_id in record_ids:
            votes += self.rasterize_record(record_id).astype(np.uint8, copy=False)
        return votes, len(record_ids)


def aggregate_votes(
    vote_counts: np.ndarray,
    annotator_count: int,
    *,
    target_mode: TargetMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a training target and a fixed majority-vote diagnostic target."""

    if target_mode not in TARGET_MODES:
        raise M2DataError(f"unknown target mode: {target_mode!r}")
    votes = np.asarray(vote_counts)
    if votes.ndim != 2 or not np.issubdtype(votes.dtype, np.integer):
        raise M2DataError("vote_counts must be a two-dimensional integer array")
    if annotator_count < 1 or np.any(votes < 0) or np.any(votes > annotator_count):
        raise M2DataError("vote_counts fall outside the annotator range")
    vote_fraction = votes.astype(np.float32) / np.float32(annotator_count)
    majority = np.ascontiguousarray(vote_fraction >= 0.5, dtype=np.uint8)
    if target_mode == "soft_vote":
        target = vote_fraction
    elif target_mode == "majority":
        target = majority.astype(np.float32)
    else:
        target = (votes > 0).astype(np.float32)
    return np.ascontiguousarray(target), majority


def percentile_normalize(
    image: np.ndarray,
    *,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    """Clip a grayscale image to two percentiles and map it to float32 [0, 1]."""

    array = np.asarray(image)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise M2DataError("grayscale image must be a two-dimensional numeric array")
    if not 0 <= lower < upper <= 100:
        raise M2DataError("normalization percentiles must satisfy 0 <= lower < upper <= 100")
    lower_value, upper_value = np.percentile(array, (lower, upper))
    if not np.isfinite(lower_value) or not np.isfinite(upper_value):
        raise M2DataError("normalization percentiles must be finite")
    if upper_value <= lower_value:
        return np.zeros(array.shape, dtype=np.float32)
    normalized = (array.astype(np.float32) - np.float32(lower_value)) / np.float32(
        upper_value - lower_value
    )
    return np.ascontiguousarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)


def _normalise_size(
    value: int | tuple[int, int] | None,
    *,
    name: str,
) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, tuple) and len(value) == 2:
        result = value
    else:
        raise M2DataError(f"{name} must be an integer, (height, width), or None")
    if any(not isinstance(side, int) or side <= 0 for side in result):
        raise M2DataError(f"{name} dimensions must be positive integers")
    return result


def _axis_origins(length: int, window: int, stride: int) -> tuple[int, ...]:
    if window > length:
        raise M2DataError(f"patch dimension {window} exceeds source dimension {length}")
    origins = list(range(0, length - window + 1, stride))
    final_origin = length - window
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return tuple(origins)


def _resize(array: np.ndarray, size: tuple[int, int], *, nearest: bool) -> np.ndarray:
    height, width = size
    source_dtype = np.float32 if np.issubdtype(array.dtype, np.floating) else np.uint8
    source = Image.fromarray(array.astype(source_dtype, copy=False))
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    resized = source.resize((width, height), resample=resample)
    return np.asarray(resized).copy()


class M2SegmentationDataset:
    """A NumPy map-style dataset accepted directly by ``torch.utils.data.DataLoader``."""

    def __init__(
        self,
        data_root: Path,
        *,
        split_manifest: Path = Path("artifacts/m1/split_manifest.csv"),
        annotation_path: Path | None = None,
        split: DatasetSplit,
        fold: int,
        image_size: int | tuple[int, int] | None = 512,
        target_mode: TargetMode = "soft_vote",
        patch_size: int | tuple[int, int] | None = None,
        patch_stride: int | tuple[int, int] | None = None,
        normalization_percentiles: tuple[float, float] = (1.0, 99.0),
        expected_manifest_sha256: str | None = FROZEN_M1_SPLIT_MANIFEST_SHA256,
        target_cache_root: Path | None = None,
        memory_cache_size: int = 2,
        expected_annotation_sha256: str | None = FROZEN_ANNOTATION_SHA256,
        verify_image_hashes: bool = True,
        input_feature_mode: str = "none",
        local_contrast_blur_radius: float = 31.0,
        local_contrast_percentiles: tuple[float, float] = (1.0, 99.0),
        local_contrast_bounds: tuple[float, float] | None = None,
        feature_cache_dir: Path | None = None,
        config: dict | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.split = split
        self.fold = fold
        self.image_size = _normalise_size(image_size, name="image_size")
        self.target_mode = target_mode
        if target_mode not in TARGET_MODES:
            raise M2DataError(f"unknown target mode: {target_mode!r}")
        self.patch_size = _normalise_size(patch_size, name="patch_size")
        if patch_stride is not None and self.patch_size is None:
            raise M2DataError("patch_stride requires patch_size")
        self.patch_stride = _normalise_size(
            patch_stride if patch_stride is not None else self.patch_size,
            name="patch_stride",
        )
        if (
            not isinstance(normalization_percentiles, tuple)
            or len(normalization_percentiles) != 2
        ):
            raise M2DataError("normalization_percentiles must be a (lower, upper) tuple")
        self.normalization_percentiles = normalization_percentiles
        if input_feature_mode not in {"none", "local_contrast_residual"}:
            raise M2DataError(f"unknown input_feature_mode: {input_feature_mode!r}")
        if (
            not isinstance(local_contrast_percentiles, tuple)
            or len(local_contrast_percentiles) != 2
        ):
            raise M2DataError("local_contrast_percentiles must be a (lower, upper) tuple")
        self.input_feature_mode = input_feature_mode
        self.local_contrast_config = LocalContrastFeatureConfig(
            blur_radius=float(local_contrast_blur_radius),
            lower_percentile=float(local_contrast_percentiles[0]),
            upper_percentile=float(local_contrast_percentiles[1]),
        )
        try:
            self.local_contrast_config.validate()
        except ValueError as error:
            raise M2DataError(str(error)) from error
        if local_contrast_bounds is not None:
            if (
                not isinstance(local_contrast_bounds, tuple)
                or len(local_contrast_bounds) != 2
                or not all(np.isfinite(value) for value in local_contrast_bounds)
                or float(local_contrast_bounds[1]) <= float(local_contrast_bounds[0])
            ):
                raise M2DataError(
                    "local_contrast_bounds must be a finite (lower, upper) tuple with lower < upper"
                )
            self.local_contrast_bounds = (
                float(local_contrast_bounds[0]),
                float(local_contrast_bounds[1]),
            )
        else:
            self.local_contrast_bounds = None
        if not isinstance(memory_cache_size, int) or memory_cache_size < 0:
            raise M2DataError("memory_cache_size must be a non-negative integer")
        self.memory_cache_size = memory_cache_size
        self.feature_cache_index = None
        if feature_cache_dir is not None:
            from .m10_feature_cache import M10FeatureCacheError, load_feature_cache_index

            try:
                self.feature_cache_index = load_feature_cache_index(
                    feature_cache_dir, fold=fold
                )
            except M10FeatureCacheError as error:
                raise M2DataError(str(error)) from error

        manifest_records = load_split_manifest(
            split_manifest,
            expected_sha256=expected_manifest_sha256,
        )
        self.records = select_fold_records(manifest_records, split=split, fold=fold)
        annotation_path = annotation_path or (
            self.data_root / "train" / ANNOTATION_FILE
        )
        self.annotation_index = CocoSemanticIndex(annotation_path)
        if (
            expected_annotation_sha256 is not None
            and self.annotation_index.annotation_sha256 != expected_annotation_sha256
        ):
            raise M2DataError(
                "annotation JSON hash differs from the frozen M0 artifact: "
                f"expected {expected_annotation_sha256}, "
                f"got {self.annotation_index.annotation_sha256}"
            )
        self.annotation_index.validate_manifest(manifest_records)
        self.image_dir = self.data_root / "train" / "train_images"
        if not self.image_dir.is_dir():
            raise M2DataError(f"training image directory does not exist: {self.image_dir}")
        for record in self.records:
            image_path = self.image_dir / record.file_name
            if not image_path.is_file():
                raise M2DataError(f"training image is missing: {record.file_name}")
            if verify_image_hashes and _sha256(image_path) != record.image_sha256:
                raise M2DataError(f"training image hash differs for {record.file_name}")

        self.target_cache_dir = (
            None
            if target_cache_root is None
            else Path(target_cache_root).resolve() / target_mode
        )
        self._vote_cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.samples = self._build_samples()

        # M12 augmentation integration
        self.m12_augmentation = None
        if config is not None:
            self.m12_augmentation = build_augmentation_from_config(config)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(record.image_id for record in self.records)

    def _build_samples(self) -> tuple[SampleWindow, ...]:
        samples: list[SampleWindow] = []
        for record in self.records:
            source_height, source_width = self.annotation_index.dimensions_by_image[
                record.image_id
            ]
            if self.patch_size is None:
                samples.append(
                    SampleWindow(record, 0, 0, source_height, source_width)
                )
                continue
            patch_height, patch_width = self.patch_size
            assert self.patch_stride is not None
            stride_height, stride_width = self.patch_stride
            for top in _axis_origins(source_height, patch_height, stride_height):
                for left in _axis_origins(source_width, patch_width, stride_width):
                    samples.append(
                        SampleWindow(record, top, left, patch_height, patch_width)
                    )
        if not samples:
            raise M2DataError("dataset contains no samples")
        return tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _remember(
        self,
        cache: OrderedDict[str, object],
        key: str,
        value: object,
    ) -> object:
        if self.memory_cache_size == 0:
            return value
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.memory_cache_size:
            cache.popitem(last=False)
        return value

    def _load_image(self, record: SplitRecord) -> np.ndarray:
        cached = self._image_cache.get(record.image_id)
        if cached is not None:
            self._image_cache.move_to_end(record.image_id)
            return cached
        path = self.image_dir / record.file_name
        with Image.open(path) as image:
            if image.mode != "L":
                raise M2DataError(f"source image is not grayscale L: {path}")
            array = np.asarray(image).copy()
        expected_shape = self.annotation_index.dimensions_by_image[record.image_id]
        if array.shape != expected_shape:
            raise M2DataError(
                f"source image shape differs from COCO for {record.image_id}: {array.shape}"
            )
        normalized = percentile_normalize(
            array,
            lower=float(self.normalization_percentiles[0]),
            upper=float(self.normalization_percentiles[1]),
        )
        return self._remember(
            self._image_cache, record.image_id, normalized
        )  # type: ignore[return-value]

    def _cache_path(self, image_id: str) -> Path | None:
        if self.target_cache_dir is None:
            return None
        return self.target_cache_dir / f"{image_id}.npz"

    def _read_cached_votes(self, path: Path, image_id: str) -> tuple[np.ndarray, int]:
        try:
            with np.load(path, allow_pickle=False) as cached:
                votes = np.asarray(cached["vote_counts"], dtype=np.uint8).copy()
                annotator_count = int(cached["annotator_count"])
                annotation_sha256 = str(cached["annotation_sha256"])
        except (OSError, ValueError, KeyError) as error:
            raise M2DataError(f"invalid target cache file: {path}") from error
        if annotation_sha256 != self.annotation_index.annotation_sha256:
            raise M2DataError(f"stale target cache for {image_id}: annotation hash differs")
        expected_shape = self.annotation_index.dimensions_by_image[image_id]
        expected_annotators = len(self.annotation_index.record_ids_by_image[image_id])
        if votes.shape != expected_shape or annotator_count != expected_annotators:
            raise M2DataError(f"target cache contract mismatch for {image_id}")
        return votes, annotator_count

    def _load_votes(self, image_id: str) -> tuple[np.ndarray, int]:
        cached = self._vote_cache.get(image_id)
        if cached is not None:
            self._vote_cache.move_to_end(image_id)
            return cached
        cache_path = self._cache_path(image_id)
        if cache_path is not None and cache_path.is_file():
            result = self._read_cached_votes(cache_path, image_id)
        else:
            result = self.annotation_index.vote_counts(image_id)
        return self._remember(
            self._vote_cache, image_id, result
        )  # type: ignore[return-value]

    def _write_cached_votes(
        self,
        image_id: str,
        votes: np.ndarray,
        annotator_count: int,
    ) -> Path:
        cache_path = self._cache_path(image_id)
        if cache_path is None:
            raise M2DataError("target_cache_root is required to build the target cache")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp.npz")
        majority = (votes.astype(np.float32) / annotator_count >= 0.5).astype(np.uint8)
        np.savez_compressed(
            temporary_path,
            vote_counts=votes,
            majority=majority,
            annotator_count=np.asarray(annotator_count, dtype=np.uint8),
            annotation_sha256=np.asarray(self.annotation_index.annotation_sha256),
        )
        temporary_path.replace(cache_path)
        return cache_path

    def build_target_cache(self, *, overwrite: bool = False) -> tuple[Path, ...]:
        """Materialize selected physical targets before starting DataLoader workers."""

        if self.target_cache_dir is None:
            raise M2DataError("target_cache_root is required to build the target cache")
        paths: list[Path] = []
        for record in self.records:
            cache_path = self._cache_path(record.image_id)
            assert cache_path is not None
            if cache_path.is_file() and not overwrite:
                self._read_cached_votes(cache_path, record.image_id)
                paths.append(cache_path)
                continue
            votes, annotator_count = self.annotation_index.vote_counts(record.image_id)
            paths.append(
                self._write_cached_votes(record.image_id, votes, annotator_count)
            )
        return tuple(paths)

    def _feature_stack(self, image_id: str, image: np.ndarray) -> np.ndarray:
        """Return the runtime input stack, preferring a verified feature cache."""

        if self.feature_cache_index is not None:
            return self.feature_cache_index.load(image_id)
        return build_feature_stack(
            image,
            feature_mode=self.input_feature_mode,
            local_contrast_config=self.local_contrast_config,
            local_contrast_bounds=self.local_contrast_bounds,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        record = sample.record
        image = self._load_image(record)
        votes, annotator_count = self._load_votes(record.image_id)
        target, binary_target = aggregate_votes(
            votes, annotator_count, target_mode=self.target_mode
        )

        row_slice = slice(sample.top, sample.top + sample.height)
        column_slice = slice(sample.left, sample.left + sample.width)
        image = image[row_slice, column_slice]
        target = target[row_slice, column_slice]
        binary_target = binary_target[row_slice, column_slice]
        if self.image_size is not None and image.shape != self.image_size:
            image = _resize(image, self.image_size, nearest=False)
            target = _resize(target, self.image_size, nearest=True)
            binary_target = _resize(binary_target, self.image_size, nearest=True)

        # Apply M12 augmentation after resize but before feature stack
        # Only augment training split to avoid data leakage
        if self.m12_augmentation is not None and self.split == "train":
            image, target = apply_m12_augmentation(
                image,
                target.astype(np.uint8),
                self.m12_augmentation
            )
            # Ensure target is back to float32
            target = target.astype(np.float32)
            # Update binary target from augmented target
            binary_target = (target >= 0.5).astype(np.uint8)

        return {
            "image_id": record.image_id,
            "file_name": record.file_name,
            "image": self._feature_stack(record.image_id, image),
            "target": np.ascontiguousarray(target[None, ...], dtype=np.float32),
            "binary_target": np.ascontiguousarray(
                binary_target[None, ...], dtype=np.uint8
            ),
            "annotator_count": annotator_count,
            "fold": -1 if record.fold is None else record.fold,
            "role": record.role,
            "crop_box": np.asarray(
                [sample.left, sample.top, sample.width, sample.height], dtype=np.int32
            ),
        }


class M2TestDataset:
    """Deterministically load normalized competition test images for inference."""

    def __init__(
        self,
        data_root: Path,
        *,
        image_size: int | tuple[int, int] | None = 512,
        normalization_percentiles: tuple[float, float] = (1.0, 99.0),
        image_manifest: Path = Path("artifacts/m0/images.csv"),
        expected_manifest_sha256: str | None = FROZEN_M0_IMAGE_MANIFEST_SHA256,
        input_feature_mode: str = "none",
        local_contrast_blur_radius: float = 31.0,
        local_contrast_percentiles: tuple[float, float] = (1.0, 99.0),
        local_contrast_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.image_dir = self.data_root / "test" / "test_images"
        if not self.image_dir.is_dir():
            raise M2DataError(f"test image directory does not exist: {self.image_dir}")
        self.image_size = _normalise_size(image_size, name="image_size")
        if (
            not isinstance(normalization_percentiles, tuple)
            or len(normalization_percentiles) != 2
        ):
            raise M2DataError("normalization_percentiles must be a (lower, upper) tuple")
        self.normalization_percentiles = normalization_percentiles
        if input_feature_mode not in {"none", "local_contrast_residual"}:
            raise M2DataError(f"unknown input_feature_mode: {input_feature_mode!r}")
        if (
            not isinstance(local_contrast_percentiles, tuple)
            or len(local_contrast_percentiles) != 2
        ):
            raise M2DataError("local_contrast_percentiles must be a (lower, upper) tuple")
        self.input_feature_mode = input_feature_mode
        self.local_contrast_config = LocalContrastFeatureConfig(
            blur_radius=float(local_contrast_blur_radius),
            lower_percentile=float(local_contrast_percentiles[0]),
            upper_percentile=float(local_contrast_percentiles[1]),
        )
        try:
            self.local_contrast_config.validate()
        except ValueError as error:
            raise M2DataError(str(error)) from error
        if local_contrast_bounds is not None:
            if (
                not isinstance(local_contrast_bounds, tuple)
                or len(local_contrast_bounds) != 2
                or not all(np.isfinite(value) for value in local_contrast_bounds)
                or float(local_contrast_bounds[1]) <= float(local_contrast_bounds[0])
            ):
                raise M2DataError(
                    "local_contrast_bounds must be a finite (lower, upper) tuple with lower < upper"
                )
            self.local_contrast_bounds = (
                float(local_contrast_bounds[0]),
                float(local_contrast_bounds[1]),
            )
        else:
            self.local_contrast_bounds = None
        self.paths = tuple(sorted(path for path in self.image_dir.iterdir() if path.is_file()))
        if not self.paths:
            raise M2DataError("test image directory must not be empty")
        invalid = [path.name for path in self.paths if IMAGE_NAME_PATTERN.fullmatch(path.name) is None]
        if invalid:
            raise M2DataError(f"invalid test image file names: {invalid[:3]}")
        manifest_path = Path(image_manifest).resolve()
        if not manifest_path.is_file():
            raise M2DataError(f"M0 image manifest does not exist: {manifest_path}")
        if expected_manifest_sha256 is not None:
            actual_hash = _sha256(manifest_path)
            if actual_hash != expected_manifest_sha256:
                raise M2DataError(
                    "M0 image manifest hash differs from the frozen artifact: "
                    f"expected {expected_manifest_sha256}, got {actual_hash}"
                )
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"split", "image_id", "file_name", "sha256"}.issubset(
                reader.fieldnames or ()
            ):
                raise M2DataError("M0 image manifest lacks test identity columns")
            expected = {
                row["image_id"]: (row["file_name"], row["sha256"].lower())
                for row in reader
                if row["split"] == "test"
            }
        actual_names = {path.stem: path.name for path in self.paths}
        expected_names = {image_id: values[0] for image_id, values in expected.items()}
        if not expected or actual_names != expected_names:
            raise M2DataError(
                "test files differ from the frozen M0 manifest: "
                f"missing={sorted(set(expected) - set(actual_names))[:3]}, "
                f"extra={sorted(set(actual_names) - set(expected))[:3]}"
            )
        for path in self.paths:
            if _sha256(path) != expected[path.stem][1]:
                raise M2DataError(f"test image hash differs for {path.name}")

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(path.stem for path in self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        with Image.open(path) as image:
            if image.mode != "L":
                raise M2DataError(f"source image is not grayscale L: {path}")
            array = np.asarray(image).copy()
        if array.shape != IMAGE_SHAPE:
            raise M2DataError(f"unexpected test image shape for {path.name}: {array.shape}")
        normalized = percentile_normalize(
            array,
            lower=float(self.normalization_percentiles[0]),
            upper=float(self.normalization_percentiles[1]),
        )
        if self.image_size is not None and normalized.shape != self.image_size:
            normalized = _resize(normalized, self.image_size, nearest=False)
        return {
            "image_id": path.stem,
            "file_name": path.name,
            "image": build_feature_stack(
                normalized,
                feature_mode=self.input_feature_mode,
                local_contrast_config=self.local_contrast_config,
                local_contrast_bounds=self.local_contrast_bounds,
            ),
        }


def build_target_cache(
    data_root: Path,
    *,
    split_manifest: Path = Path("artifacts/m1/split_manifest.csv"),
    annotation_path: Path | None = None,
    output_root: Path = Path("artifacts/m2/cache/targets"),
    target_mode: TargetMode = "soft_vote",
    expected_manifest_sha256: str | None = FROZEN_M1_SPLIT_MANIFEST_SHA256,
    overwrite: bool = False,
    expected_annotation_sha256: str | None = FROZEN_ANNOTATION_SHA256,
) -> tuple[Path, ...]:
    """Build target caches for every physical image exactly once."""

    data_root = Path(data_root).resolve()
    records = load_split_manifest(
        split_manifest,
        expected_sha256=expected_manifest_sha256,
    )
    annotation_path = annotation_path or data_root / "train" / ANNOTATION_FILE
    index = CocoSemanticIndex(annotation_path)
    if (
        expected_annotation_sha256 is not None
        and index.annotation_sha256 != expected_annotation_sha256
    ):
        raise M2DataError(
            "annotation JSON hash differs from the frozen M0 artifact: "
            f"expected {expected_annotation_sha256}, got {index.annotation_sha256}"
        )
    index.validate_manifest(records)
    output_dir = Path(output_root).resolve() / target_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for record in records:
        cache_path = output_dir / f"{record.image_id}.npz"
        if cache_path.is_file() and not overwrite:
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    votes = np.asarray(cached["vote_counts"], dtype=np.uint8)
                    annotator_count = int(cached["annotator_count"])
                    annotation_sha256 = str(cached["annotation_sha256"])
            except (OSError, ValueError, KeyError) as error:
                raise M2DataError(f"invalid target cache file: {cache_path}") from error
            if (
                annotation_sha256 != index.annotation_sha256
                or votes.shape != index.dimensions_by_image[record.image_id]
                or annotator_count != len(index.record_ids_by_image[record.image_id])
                or np.any(votes > annotator_count)
            ):
                raise M2DataError(f"target cache contract mismatch for {record.image_id}")
            paths.append(cache_path)
            continue
        votes, annotator_count = index.vote_counts(record.image_id)
        temporary_path = cache_path.with_suffix(".tmp.npz")
        majority = (votes.astype(np.float32) / annotator_count >= 0.5).astype(np.uint8)
        np.savez_compressed(
            temporary_path,
            vote_counts=votes,
            majority=majority,
            annotator_count=np.asarray(annotator_count, dtype=np.uint8),
            annotation_sha256=np.asarray(index.annotation_sha256),
        )
        temporary_path.replace(cache_path)
        paths.append(cache_path)
    return tuple(paths)


__all__ = [
    "CocoSemanticIndex",
    "FROZEN_M0_IMAGE_MANIFEST_SHA256",
    "FROZEN_ANNOTATION_SHA256",
    "FROZEN_M1_SPLIT_MANIFEST_SHA256",
    "M2DataError",
    "M2SegmentationDataset",
    "M2TestDataset",
    "SplitRecord",
    "aggregate_votes",
    "build_target_cache",
    "load_split_manifest",
    "percentile_normalize",
    "runtime_image_array",
    "select_fold_records",
]
