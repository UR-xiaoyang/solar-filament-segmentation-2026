"""Validate the downloaded competition data and freeze its published contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .dataset_archive import ArchiveInspector, inspect_archive
from .dataset_audit import DatasetAuditor, audit_dataset
from .dataset_coco import (
    AnnotationSource,
    AnnotationValidation,
    AnnotationValidator,
    _load_annotations,
    _validate_polygon,
)
from .dataset_contract import (
    ANNOTATION_FILE,
    ANNOTATION_RECORD_COLUMNS,
    COMPETITION_SLUG,
    DATASET_DIRECTORY,
    DatasetContractError,
    IMAGE_MANIFEST_COLUMNS,
    IMAGE_NAME_PATTERN,
    _distribution,
    _require,
    sha256_file,
)
from .dataset_images import ImageScanner, _scan_image
from .dataset_manifest import ManifestBuilder, ManifestWriter, _write_csv, write_audit_artifacts

__all__ = [
    "ANNOTATION_FILE",
    "ANNOTATION_RECORD_COLUMNS",
    "COMPETITION_SLUG",
    "DATASET_DIRECTORY",
    "DatasetContractError",
    "IMAGE_MANIFEST_COLUMNS",
    "IMAGE_NAME_PATTERN",
    "audit_dataset",
    "inspect_archive",
    "main",
    "sha256_file",
    "write_audit_artifacts",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m0"))
    args = parser.parse_args(argv)

    contract, image_manifest, annotation_manifest = audit_dataset(
        args.data_root,
        archive_path=args.archive,
    )
    write_audit_artifacts(args.output, contract, image_manifest, annotation_manifest)
    print(json.dumps(contract["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
