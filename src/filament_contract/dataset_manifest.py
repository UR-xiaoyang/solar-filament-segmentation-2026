"""Build and write the normalized M0 image and annotation-record manifests."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contract import IMAGE_SHAPE
from .dataset_contract import (
    ANNOTATION_RECORD_COLUMNS,
    IMAGE_MANIFEST_COLUMNS,
    IMAGE_NAME_PATTERN,
    _require,
    sha256_file,
)
from .dataset_images import _scan_image


class ManifestBuilder:
    """Build the normalized image and annotation-record manifests."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def build_annotation_manifest(
        self,
        image_records: Sequence[Mapping[str, Any]],
        instances_by_record: Mapping[str, int],
    ) -> tuple[list[dict[str, object]], dict[str, list[dict[str, Any]]]]:
        records_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        annotation_manifest: list[dict[str, object]] = []
        for record in sorted(image_records, key=lambda item: item["id"]):
            file_name = record.get("file_name")
            match = IMAGE_NAME_PATTERN.fullmatch(file_name)
            _require(match is not None, f"invalid train file name: {file_name}")
            _require(record.get("width") == IMAGE_SHAPE[1], f"invalid JSON width: {file_name}")
            _require(record.get("height") == IMAGE_SHAPE[0], f"invalid JSON height: {file_name}")
            image_id = file_name.removesuffix(".jpeg")
            image_record_id = record["id"]
            prefix, separator, suffix = image_record_id.partition("-")
            _require(
                separator == "-" and suffix == image_id, f"invalid image record ID: {image_record_id}"
            )
            records_by_file[file_name].append(record)
            annotation_manifest.append(
                {
                    "image_record_id": image_record_id,
                    "image_id": image_id,
                    "file_name": file_name,
                    "annotator_batch": prefix,
                    "date_captured": record.get("date_captured", ""),
                    "instance_count": instances_by_record[image_record_id],
                }
            )
        return annotation_manifest, records_by_file

    def build_image_manifest(
        self,
        splits: Sequence[tuple[str, Sequence[Path]]],
        records_by_file: Mapping[str, Sequence[Mapping[str, Any]]],
        instances_by_record: Mapping[str, int],
    ) -> tuple[list[dict[str, object]], list[list[str]]]:
        image_manifest: list[dict[str, object]] = []
        image_hashes: defaultdict[str, list[str]] = defaultdict(list)
        for split, paths in splits:
            for path in paths:
                match = IMAGE_NAME_PATTERN.fullmatch(path.name)
                _require(match is not None, f"invalid {split} file name: {path.name}")
                width, height, mode, image_format = _scan_image(path)
                digest = sha256_file(path)
                image_hashes[digest].append(f"{split}/{path.name}")
                source_records = records_by_file.get(path.name, [])
                image_manifest.append(
                    {
                        "split": split,
                        "image_id": path.stem,
                        "file_name": path.name,
                        "relative_path": path.relative_to(self.data_root).as_posix(),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "format": image_format,
                        "size_bytes": path.stat().st_size,
                        "sha256": digest,
                        "annotation_record_count": len(source_records),
                        "instance_count": sum(
                            instances_by_record[record["id"]] for record in source_records
                        ),
                    }
                )
        duplicates = [paths for paths in image_hashes.values() if len(paths) > 1]
        return image_manifest, duplicates


class ManifestWriter:
    """Write the audit contract and normalized manifests as M0 artifacts."""

    def write(
        self,
        output_dir: Path,
        contract: Mapping[str, Any],
        image_manifest: Sequence[Mapping[str, object]],
        annotation_manifest: Sequence[Mapping[str, object]],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = output_dir / "data_contract.json"
        with contract_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        _write_csv(output_dir / "images.csv", IMAGE_MANIFEST_COLUMNS, image_manifest)
        _write_csv(
            output_dir / "annotation_records.csv",
            ANNOTATION_RECORD_COLUMNS,
            annotation_manifest,
        )


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


_WRITER = ManifestWriter()


def write_audit_artifacts(
    output_dir: Path,
    contract: Mapping[str, Any],
    image_manifest: Sequence[Mapping[str, object]],
    annotation_manifest: Sequence[Mapping[str, object]],
) -> None:
    _WRITER.write(output_dir, contract, image_manifest, annotation_manifest)


__all__ = ["ManifestBuilder", "ManifestWriter", "write_audit_artifacts"]
