"""Regression coverage for 256×256 candidate patch extraction."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "quality_judge_patch256_precompute.py"
    spec = importlib.util.spec_from_file_location("patch256", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probability_is_resized_to_evaluation_shape_not_model_shape():
    module = load_module()

    probability = np.zeros((1024, 1024), dtype=np.float32)
    result = module.probability_to_evaluation_uint8(probability)

    assert result.shape == (2048, 2048)
    assert result.dtype == np.uint8


def test_edge_candidate_extracts_nonempty_patch():
    module = load_module()
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[-1, -1] = True

    patch = module.extract_patch(
        np.zeros((2048, 2048), dtype=np.uint8),
        np.zeros((2048, 2048), dtype=np.uint8),
        mask,
        patch_size=256,
        context=1.75,
    )

    assert patch.shape == (3, 256, 256)
