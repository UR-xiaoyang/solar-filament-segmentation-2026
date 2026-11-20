"""Contract tests for the FS-1 fast-screen route registry.

The registry is config-as-data that drives GPU experiments, so a typo in a
config key or a missing patch module must fail here instead of burning a
25-minute training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from fastscreen_routes import FROZEN_CONTROL, ROUTES  # noqa: E402

DEVICES = {"cuda:0", "cuda:1", "cuda:2"}


def test_route_ids_unique_and_well_formed() -> None:
    ids = [route["id"] for route in ROUTES]
    assert len(ids) == len(set(ids))
    assert all(route_id.startswith("F") and route_id[1:].isdigit() for route_id in ids)


def test_every_route_declares_exactly_one_mechanism_delta() -> None:
    for route in ROUTES:
        config = route.get("config") or {}
        patch = route.get("patch")
        assert config or patch, f"{route['id']} declares no mechanism"
        assert route["mechanism"].strip().endswith("."), route["id"]
        for key in config:
            assert key in FROZEN_CONTROL, f"{route['id']} overrides unknown field {key}"
            if key not in (route.get("allow_diff") or []):
                pytest.fail(f"{route['id']} changes {key} without declaring allow_diff")


def test_patch_modules_exist_and_expose_apply() -> None:
    for route in ROUTES:
        patch = route.get("patch")
        if not patch:
            continue
        module_path = REPO / "scripts" / "route_patches" / f"{patch}.py"
        assert module_path.is_file(), f"{route['id']} patch module missing: {module_path}"
        namespace: dict[str, object] = {}
        exec(compile(module_path.read_text(), str(module_path), "exec"), namespace)  # noqa: S102
        assert callable(namespace.get("apply")), f"{patch} has no apply()"


def test_devices_are_known_gpus() -> None:
    for route in ROUTES:
        device = route.get("device")
        assert device is None or device in DEVICES, f"{route['id']} has device {device}"


def test_control_contract_is_the_frozen_1024_recipe() -> None:
    assert FROZEN_CONTROL["image_size"] == 1024
    assert FROZEN_CONTROL["encoder_name"] == "resnet34"
    assert FROZEN_CONTROL["epochs"] == 20
    assert FROZEN_CONTROL["seed"] == 20260713
    assert FROZEN_CONTROL["probability_threshold"] == 0.5
    assert FROZEN_CONTROL["min_instance_area"] == 256
