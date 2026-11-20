"""Regression checks for C3 topology feature joins."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import unittest


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "noise_route_c3_topology_crossfit.py"
    spec = importlib.util.spec_from_file_location("route_c3_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_join_topology_features_requires_exact_candidate_key_universe():
    module = load_module()
    candidates = [
        {"image_id": "a", "candidate_id": "0"},
        {"image_id": "b", "candidate_id": "0"},
    ]
    topology = [{"image_id": "a", "candidate_id": "0", "topo_endpoint_count": "2"}]

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "key universe"):
        module.join_topology_features(candidates, topology, ["topo_endpoint_count"])
