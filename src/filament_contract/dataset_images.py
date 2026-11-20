"""Scan physical images and verify the frozen source image contract."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .contract import IMAGE_SHAPE, SOURCE_IMAGE_FORMAT, SOURCE_IMAGE_MODE
from .dataset_contract import _require


class ImageScanner:
    """Open one image and verify its size, mode, and format."""

    def scan(self, path: Path) -> tuple[int, int, str, str]:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            image_format = image.format
        _require((height, width) == IMAGE_SHAPE, f"unexpected image size: {path}")
        _require(mode == SOURCE_IMAGE_MODE, f"unexpected image mode {mode!r}: {path}")
        _require(image_format == SOURCE_IMAGE_FORMAT, f"unexpected image format: {path}")
        return width, height, mode, image_format


_SCANNER = ImageScanner()


def _scan_image(path: Path) -> tuple[int, int, str, str]:
    return _SCANNER.scan(path)


__all__ = ["ImageScanner"]
