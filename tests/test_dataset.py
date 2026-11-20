from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from filament_contract.dataset import DatasetContractError, inspect_archive


def test_archive_inspection_checks_hash_crc_and_sizes(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/train/file.txt", b"solar")

    result = inspect_archive(archive_path)

    assert result["entry_count"] == 1
    assert result["uncompressed_bytes"] == 5
    assert result["unsafe_path_count"] == 0
    assert result["crc_check"] == "passed"
    assert len(result["sha256"]) == 64


def test_archive_inspection_rejects_parent_path(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", b"no")

    with pytest.raises(DatasetContractError, match="unsafe paths"):
        inspect_archive(archive_path)
