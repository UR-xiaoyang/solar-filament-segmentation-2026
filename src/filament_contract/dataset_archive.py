"""Inspect the downloaded competition archive for safety and integrity."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .dataset_contract import DatasetContractError, sha256_file


class ArchiveInspector:
    """Verify an archive's path safety, CRC integrity, and content digest."""

    def inspect(self, archive_path: Path) -> dict[str, Any]:
        archive_path = archive_path.resolve()
        if not archive_path.is_file():
            raise DatasetContractError(f"archive does not exist: {archive_path}")

        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            unsafe = self._unsafe_entries(entries)
            if unsafe:
                raise DatasetContractError(f"archive contains unsafe paths: {unsafe[:3]}")
            bad_entry = archive.testzip()
            if bad_entry is not None:
                raise DatasetContractError(f"archive CRC check failed: {bad_entry}")

        return {
            "file_name": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "entry_count": len(entries),
            "uncompressed_bytes": sum(entry.file_size for entry in entries),
            "unsafe_path_count": 0,
            "crc_check": "passed",
        }

    @staticmethod
    def _unsafe_entries(entries: Sequence[zipfile.ZipInfo]) -> list[str]:
        unsafe = []
        for entry in entries:
            name = entry.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts:
                unsafe.append(entry.filename)
        return unsafe


_INSPECTOR = ArchiveInspector()


def inspect_archive(archive_path: Path) -> dict[str, Any]:
    return _INSPECTOR.inspect(archive_path)


__all__ = ["ArchiveInspector", "inspect_archive"]
