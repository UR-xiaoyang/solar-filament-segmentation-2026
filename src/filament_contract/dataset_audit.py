"""Orchestrate the M0 dataset audit against the frozen competition contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contract import IMAGE_SHAPE, SOURCE_IMAGE_FORMAT, SOURCE_IMAGE_MODE
from .dataset_archive import inspect_archive
from .dataset_coco import AnnotationSource, AnnotationValidator
from .dataset_contract import (
    ANNOTATION_FILE,
    COMPETITION_SLUG,
    DATASET_DIRECTORY,
    _distribution,
    _require,
    sha256_file,
)
from .dataset_manifest import ManifestBuilder


class DatasetAuditor:
    """Validate real files and return the contract plus two normalized manifests."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()

    def audit(
        self, *, archive_path: Path | None = None
    ) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]]]:
        train_dir = self.data_root / "train" / "train_images"
        test_dir = self.data_root / "test" / "test_images"
        annotation_path = self.data_root / "train" / ANNOTATION_FILE
        for required_path in (train_dir, test_dir, annotation_path):
            _require(required_path.exists(), f"required dataset path is missing: {required_path}")

        train_paths = sorted(path for path in train_dir.iterdir() if path.is_file())
        test_paths = sorted(path for path in test_dir.iterdir() if path.is_file())
        _require(train_paths and test_paths, "train and test image directories must not be empty")
        _require(
            all(path.suffix == ".jpeg" for path in train_paths + test_paths),
            "all competition images must use the .jpeg extension",
        )

        annotations_json = AnnotationSource().load(annotation_path)
        validation = AnnotationValidator().validate(annotations_json)
        image_records = validation.image_records
        annotations = validation.annotations
        categories = validation.categories

        train_names = {path.name for path in train_paths}
        test_names = {path.name for path in test_paths}
        json_names = {record.get("file_name") for record in image_records}
        _require(train_names == json_names, "physical train images and JSON file names differ")
        _require(not train_names.intersection(test_names), "train and test file names overlap")

        builder = ManifestBuilder(self.data_root)
        annotation_manifest, records_by_file = builder.build_annotation_manifest(
            image_records, validation.instances_by_record
        )
        image_manifest, duplicates = builder.build_image_manifest(
            (("train", train_paths), ("test", test_paths)),
            records_by_file,
            validation.instances_by_record,
        )

        records_per_file = Counter(len(records) for records in records_by_file.values())
        instances_per_record = list(validation.instances_by_record.values())
        category_usage = Counter(annotation["category_id"] for annotation in annotations)
        archive_info = inspect_archive(archive_path) if archive_path is not None else None

        contract: dict[str, Any] = {
            "schema_version": 1,
            "competition_slug": COMPETITION_SLUG,
            "dataset_directory": DATASET_DIRECTORY,
            "license": "CC BY-NC 4.0",
            "archive": archive_info,
            "annotation_file": {
                "file_name": annotation_path.name,
                "size_bytes": annotation_path.stat().st_size,
                "sha256": sha256_file(annotation_path),
                "segmentation_representation": "single COCO polygon per instance",
                "licenses_representation": (
                    "object" if isinstance(annotations_json["licenses"], dict) else "list"
                ),
            },
            "image_contract": {
                "height": IMAGE_SHAPE[0],
                "width": IMAGE_SHAPE[1],
                "mode": SOURCE_IMAGE_MODE,
                "format": SOURCE_IMAGE_FORMAT,
                "extension": ".jpeg",
                "file_name_pattern": "YYYYMMDDHHMMSSII.jpeg",
            },
            "counts": {
                "archive_entries": 0 if archive_info is None else archive_info["entry_count"],
                "train_physical_images": len(train_paths),
                "test_physical_images": len(test_paths),
                "annotation_image_records": len(image_records),
                "unique_annotated_file_names": len(json_names),
                "annotation_instances": len(annotations),
                "categories_declared": len(categories),
                "categories_used": len(category_usage),
                "explicitly_closed_polygons": validation.explicitly_closed,
                "implicitly_closed_polygons": len(annotations) - validation.explicitly_closed,
                "exact_duplicate_image_groups": len(duplicates),
            },
            "distributions": {
                "annotation_records_per_physical_image": {
                    str(key): value for key, value in sorted(records_per_file.items())
                },
                "instances_per_annotation_record": {
                    "minimum": min(instances_per_record),
                    "maximum": max(instances_per_record),
                    "mean": sum(instances_per_record) / len(instances_per_record),
                    "histogram": _distribution(instances_per_record),
                },
                "category_instances": {
                    str(category_id): category_usage.get(category_id, 0)
                    for category_id in sorted(validation.category_ids)
                },
            },
            "integrity": {
                "archive_crc": "not_checked" if archive_path is None else "passed",
                "unsafe_archive_paths": 0,
                "unknown_annotation_image_references": 0,
                "duplicate_annotation_ids": 0,
                "unannotated_image_records": 0,
                "json_file_names_missing_on_disk": 0,
                "unreferenced_train_images": 0,
                "train_test_file_name_overlap": 0,
                "non_polygon_segmentations": 0,
                "invalid_polygon_coordinates": 0,
                "invalid_bboxes": 0,
                "nonpositive_areas": 0,
            },
            "exact_duplicate_image_groups": duplicates,
        }
        return contract, image_manifest, annotation_manifest


def audit_dataset(
    data_root: Path,
    *,
    archive_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]]]:
    """Validate real files and return the contract plus two normalized manifests."""

    return DatasetAuditor(data_root).audit(archive_path=archive_path)


__all__ = ["DatasetAuditor", "audit_dataset"]
