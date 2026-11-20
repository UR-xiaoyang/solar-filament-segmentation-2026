"""Load and validate the COCO annotation source of the competition."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import IMAGE_SHAPE
from .dataset_contract import _require


@dataclass(frozen=True)
class AnnotationValidation:
    """Validated COCO record sets and per-record instance counts."""

    image_records: Sequence[Mapping[str, Any]]
    annotations: Sequence[Mapping[str, Any]]
    categories: Sequence[Mapping[str, Any]]
    records_by_id: dict[str, Mapping[str, Any]]
    category_ids: set[Any]
    instances_by_record: Counter[str]
    explicitly_closed: int


class AnnotationSource:
    """Load and structurally validate the COCO annotation JSON."""

    def load(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        _require(isinstance(data, dict), "annotation root must be a JSON object")
        for key in ("images", "annotations", "categories"):
            _require(
                isinstance(data.get(key), list), f"annotation field {key!r} must be a list"
            )
        _require(
            isinstance(data.get("licenses"), (dict, list)),
            "annotation field 'licenses' must be an object or list",
        )
        return data


class AnnotationValidator:
    """Validate annotation records and derive per-image instance counts."""

    def validate(self, annotations_json: Mapping[str, Any]) -> AnnotationValidation:
        image_records = annotations_json["images"]
        annotations = annotations_json["annotations"]
        categories = annotations_json["categories"]
        records_by_id, category_ids = self._validate_records(
            image_records, annotations, categories
        )
        instances_by_record, explicitly_closed = self._count_instances(
            annotations, records_by_id, category_ids
        )
        _require(
            set(instances_by_record) == set(records_by_id),
            "every annotation image record must contain at least one instance",
        )
        return AnnotationValidation(
            image_records,
            annotations,
            categories,
            records_by_id,
            category_ids,
            instances_by_record,
            explicitly_closed,
        )

    @staticmethod
    def _validate_records(
        image_records: Sequence[Mapping[str, Any]],
        annotations: Sequence[Mapping[str, Any]],
        categories: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Mapping[str, Any]], set[Any]]:
        image_record_ids = [record.get("id") for record in image_records]
        _require(
            all(isinstance(value, str) and value for value in image_record_ids),
            "every annotation image record must have a string ID",
        )
        _require(
            len(image_record_ids) == len(set(image_record_ids)),
            "annotation image record IDs must be unique",
        )
        records_by_id = {record["id"]: record for record in image_records}

        annotation_ids = [record.get("id") for record in annotations]
        _require(
            all(isinstance(value, str) and value for value in annotation_ids),
            "every annotation must have a string ID",
        )
        _require(len(annotation_ids) == len(set(annotation_ids)), "annotation IDs must be unique")

        category_ids = {record.get("id") for record in categories}
        return records_by_id, category_ids

    @staticmethod
    def _count_instances(
        annotations: Sequence[Mapping[str, Any]],
        records_by_id: Mapping[str, Mapping[str, Any]],
        category_ids: set[Any],
    ) -> tuple[Counter[str], int]:
        instances_by_record: Counter[str] = Counter()
        explicitly_closed = 0
        for annotation in annotations:
            annotation_id = annotation["id"]
            image_record_id = annotation.get("image_id")
            _require(
                image_record_id in records_by_id,
                f"annotation {annotation_id} references unknown image record",
            )
            _require(
                annotation.get("category_id") in category_ids,
                f"annotation {annotation_id} references unknown category",
            )
            _require(
                annotation.get("iscrowd") == 0, f"annotation {annotation_id} must have iscrowd=0"
            )
            _require(
                isinstance(annotation.get("area"), (int, float)) and annotation["area"] > 0,
                f"annotation {annotation_id} must have positive area",
            )
            bbox = annotation.get("bbox")
            _require(
                isinstance(bbox, list)
                and len(bbox) == 4
                and bbox[2] > 0
                and bbox[3] > 0,
                f"annotation {annotation_id} has an invalid bbox",
            )
            explicitly_closed += int(
                AnnotationValidator._validate_polygon(
                    annotation.get("segmentation"), annotation_id
                )
            )
            instances_by_record[image_record_id] += 1
        return instances_by_record, explicitly_closed

    @staticmethod
    def _validate_polygon(segmentation: object, annotation_id: str) -> bool:
        _require(
            isinstance(segmentation, list) and len(segmentation) == 1,
            f"annotation {annotation_id} must contain exactly one polygon",
        )
        polygon = segmentation[0]
        _require(
            isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0,
            f"annotation {annotation_id} has malformed polygon coordinates",
        )
        for index in range(0, len(polygon), 2):
            x, y = polygon[index], polygon[index + 1]
            _require(
                isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and math.isfinite(x)
                and math.isfinite(y),
                f"annotation {annotation_id} has non-finite polygon coordinates",
            )
            _require(
                0 <= x <= IMAGE_SHAPE[1] and 0 <= y <= IMAGE_SHAPE[0],
                f"annotation {annotation_id} has out-of-bounds polygon coordinates",
            )
        return polygon[0] == polygon[-2] and polygon[1] == polygon[-1]


_SOURCE = AnnotationSource()
_VALIDATOR = AnnotationValidator()


def _load_annotations(path: Path) -> dict[str, Any]:
    return _SOURCE.load(path)


def _validate_polygon(segmentation: object, annotation_id: str) -> bool:
    return _VALIDATOR._validate_polygon(segmentation, annotation_id)


__all__ = ["AnnotationSource", "AnnotationValidation", "AnnotationValidator"]
