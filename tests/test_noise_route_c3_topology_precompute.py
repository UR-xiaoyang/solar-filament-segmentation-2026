"""Regression tests for Route C3 topology cache features."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "noise_route_c3_topology_precompute.py"
    spec = importlib.util.spec_from_file_location("route_c3_topology", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_topology_uses_tight_crop_and_preserves_simple_line_features():
    module = load_module()
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[1000, 100:111] = True

    values = module.topology_features(mask)

    assert values["topo_skeleton_pixels"] == 11.0
    assert values["topo_endpoint_count"] == 2.0
    assert values["topo_branchpoint_count"] == 0.0
    assert values["topo_degenerate"] == 0.0


def test_topology_empty_mask_is_degenerate():
    module = load_module()

    values = module.topology_features(np.zeros((2048, 2048), dtype=bool))

    assert values["topo_skeleton_pixels"] == 0.0
    assert values["topo_degenerate"] == 1.0
