"""Tests for the A5-5 instance-aware architecture.

These tests validate the contract surface only. They do not require a
GPU or a populated data directory. Heavy smoke tests belong in the
``scripts/a5_5_maskrcnn_fold0_smoke.py`` runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from filament_contract.a5_5_instance_config import (
    A5_5_SCHEMA_VERSION,
    A5_5InstanceConfig,
    DEFAULT_FILAMENT_ANCHOR_SIZES,
    DEFAULT_FILAMENT_ASPECT_RATIOS,
    FROZEN_M1_SPLIT_MANIFEST_SHA256,
    config_from_mapping,
    read_config,
    write_config,
)
from filament_contract.a5_5_instance_data import (
    A5_5DataError,
    _strip_annotator_prefix,
    mask_to_bbox,
    polygon_to_mask,
    _hflip_bbox,
    _resize_bbox,
    _rot90_bbox,
    bbox_area,
)
from filament_contract.a5_5_filament_net import (
    EXTREME_FILAMENT_ASPECT_RATIOS,
    FilamentMaskPredictor,
    build_filament_maskrcnn,
)
from filament_contract.a5_5_instance_metrics import (
    MaskRCNNPrediction,
    _resize_mask_to_eval,
    evaluate_maskrcnn_predictions,
    maskrcnn_predictions_to_instance_masks,
)


# ---------------------------------------------------------------------------
# Config tests.
# ---------------------------------------------------------------------------


def test_a5_5_config_defaults_match_control_768_contract():
    """Default config must reuse the M2 frozen protocol for comparability."""
    config = A5_5InstanceConfig()
    assert config.schema_version == A5_5_SCHEMA_VERSION
    assert config.fold == 0
    assert config.image_size == 768
    assert config.epochs == 20
    assert config.batch_size == 8
    assert config.learning_rate == pytest.approx(3.0e-4)
    assert config.weight_decay == pytest.approx(1.0e-4)
    assert config.augment_d4 is True
    assert config.num_classes == 2
    assert config.backbone == "resnet34"
    assert config.evaluation_resolution == (2048, 2048)
    assert config.match_iou_threshold == 0.5
    assert config.min_instance_area == 256
    assert config.expected_manifest_sha256 == FROZEN_M1_SPLIT_MANIFEST_SHA256


def test_a5_5_config_anchor_defaults_are_filament_shaped():
    """Default anchors must span 1:8 .. 8:1 aspect ratios and 16..256 px widths."""
    assert min(DEFAULT_FILAMENT_ASPECT_RATIOS) <= 0.125
    assert max(DEFAULT_FILAMENT_ASPECT_RATIOS) >= 8.0
    assert min(DEFAULT_FILAMENT_ANCHOR_SIZES) <= 16
    assert max(DEFAULT_FILAMENT_ANCHOR_SIZES) >= 256


def test_a5_5_config_rejects_non_resnet34_backbone():
    with pytest.raises(ValueError, match="resnet34"):
        A5_5InstanceConfig(backbone="resnet50")


def test_a5_5_config_rejects_zero_min_area():
    with pytest.raises(ValueError, match="min_instance_area"):
        A5_5InstanceConfig(min_instance_area=0)


def test_a5_5_config_rejects_oversized_fold():
    with pytest.raises(ValueError, match="fold must be in"):
        A5_5InstanceConfig(fold=5)


def test_a5_5_config_mask_roi_pooler_default():
    config = A5_5InstanceConfig()
    assert config.mask_roi_pooler_output_size == 14


def test_a5_5_config_mask_roi_pooler_can_be_increased():
    config = A5_5InstanceConfig(mask_roi_pooler_output_size=28)
    assert config.mask_roi_pooler_output_size == 28


def test_a5_5_config_rejects_tiny_mask_roi_pooler():
    with pytest.raises(ValueError, match="mask_roi_pooler_output_size"):
        A5_5InstanceConfig(mask_roi_pooler_output_size=4)


def test_a5_5_config_with_overrides_validates_and_returns_new_instance():
    config = A5_5InstanceConfig()
    updated = config.with_overrides(fold=2, epochs=5)
    assert updated is not config
    assert updated.fold == 2
    assert updated.epochs == 5
    assert config.fold == 0  # original untouched


def test_a5_5_config_with_overrides_rejects_unknown_field():
    config = A5_5InstanceConfig()
    with pytest.raises(ValueError, match="unknown"):
        config.with_overrides(nonexistent_field=42)


def test_a5_5_config_round_trip_json(tmp_path: Path):
    config = A5_5InstanceConfig(fold=3, epochs=10)
    path = tmp_path / "cfg.json"
    write_config(path, config)
    reloaded = read_config(path)
    assert reloaded.fold == 3
    assert reloaded.epochs == 10


def test_a5_5_config_from_mapping_fills_defaults():
    config = config_from_mapping({"fold": 2})
    assert config.fold == 2
    assert config.epochs == 20  # default preserved


def test_a5_5_config_from_mapping_rejects_unknown():
    with pytest.raises(ValueError, match="unknown"):
        config_from_mapping({"foo": "bar"})


# ---------------------------------------------------------------------------
# Data utilities tests.
# ---------------------------------------------------------------------------


def _square_polygon(cx: float, cy: float, size: float) -> List[float]:
    return [
        cx - size, cy - size,
        cx + size, cy - size,
        cx + size, cy + size,
        cx - size, cy + size,
    ]


def test_polygon_to_mask_produces_correct_inside_pixels():
    mask = polygon_to_mask(_square_polygon(5, 5, 2), height=11, width=11)
    assert mask.shape == (11, 11)
    assert mask.dtype == np.uint8
    assert mask[5, 5] == 1
    assert mask[5, 3] == 1
    assert mask[5, 8] == 0


def test_polygon_to_mask_rejects_too_few_vertices():
    with pytest.raises(A5_5DataError):
        polygon_to_mask([0.0, 0.0, 1.0, 1.0], height=4, width=4)


def test_mask_to_bbox_uses_exclusive_x2_y2():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 3:8] = 1
    x1, y1, x2, y2 = mask_to_bbox(mask)
    assert (x1, y1, x2, y2) == (3.0, 2.0, 8.0, 7.0)


def test_mask_to_bbox_empty_returns_zeros():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert mask_to_bbox(mask) == (0.0, 0.0, 0.0, 0.0)


def test_bbox_area_clamps_negative():
    assert bbox_area((5.0, 5.0, 3.0, 3.0)) == 0.0
    assert bbox_area((0.0, 0.0, 4.0, 5.0)) == 20.0


def test_resize_bbox_scales_xyxy():
    bbox = (10.0, 20.0, 30.0, 40.0)
    out = _resize_bbox(bbox, original_hw=(40, 80), target_hw=(20, 40))
    assert out == pytest.approx((5.0, 10.0, 15.0, 20.0))


def test_rot90_bbox_round_trip_after_four_rotations():
    original = (10.0, 20.0, 30.0, 40.0)
    current = original
    size = 64
    # Four k=1 rotations are one full 360° turn and must restore the bbox.
    for _ in range(4):
        current = _rot90_bbox(current, (size, size), 1)
    for a, b in zip(current, original, strict=True):
        assert a == pytest.approx(b, abs=1e-4)


def test_rot90_bbox_k1_swaps_xy_for_square_image():
    # (x, y) -> (y, h - 1 - x) on a 64x64 grid.
    # bbox (x1=10, y1=20, x2=30, y2=40) becomes (y1=20, h-x2=34, y2=40, h-x1=54).
    bbox = (10.0, 20.0, 30.0, 40.0)
    rotated = _rot90_bbox(bbox, (64, 64), 1)
    assert rotated == pytest.approx((20.0, 34.0, 40.0, 54.0), abs=1e-4)


def test_rot90_bbox_k3_swaps_xy_inverse_of_k1():
    bbox = (10.0, 20.0, 30.0, 40.0)
    # (x, y) -> (h - 1 - y, x) on a 64x64 grid.
    rotated = _rot90_bbox(bbox, (64, 64), 3)
    # h - y2 = 64 - 40 = 24, x1 = 10, h - y1 = 64 - 20 = 44, x2 = 30.
    assert rotated == pytest.approx((24.0, 10.0, 44.0, 30.0), abs=1e-4)


def test_hflip_bbox_mirrors_x_within_width():
    bbox = (10.0, 20.0, 30.0, 40.0)
    flipped = _hflip_bbox(bbox, width=64)
    assert flipped == (34.0, 20.0, 54.0, 40.0)


def test_strip_annotator_prefix_removes_six_digit_prefix():
    assert _strip_annotator_prefix("010101-20110114105034Ch") == "20110114105034Ch"
    assert _strip_annotator_prefix("999999-20110114105034Ch") == "20110114105034Ch"


def test_strip_annotator_prefix_passthrough_for_unprefixed():
    assert _strip_annotator_prefix("20110114105034Ch") == "20110114105034Ch"


def test_strip_annotator_prefix_does_not_strip_non_batch_dash():
    # "abc" is not 6 digits, so we treat the dash as part of the id.
    assert _strip_annotator_prefix("abc-20110114105034Ch") == "abc-20110114105034Ch"


def test_strip_annotator_prefix_rejects_partial_batch():
    # "12345" is 5 digits, not 6 — must not strip.
    assert _strip_annotator_prefix("12345-20110114105034Ch") == "12345-20110114105034Ch"


# ---------------------------------------------------------------------------
# FilamentNet tests (WP-A5-5d specialised architecture).
# ---------------------------------------------------------------------------


def test_extreme_filament_aspect_ratios_include_thin_anchors():
    """FilamentNet must add 1:16 and 16:1 to the canonical 1:8..8:1 menu."""
    assert 0.0625 in EXTREME_FILAMENT_ASPECT_RATIOS  # 1:16
    assert 16.0 in EXTREME_FILAMENT_ASPECT_RATIOS     # 16:1
    # Still has the canonical aspect ratios.
    assert 1.0 in EXTREME_FILAMENT_ASPECT_RATIOS
    assert 8.0 in EXTREME_FILAMENT_ASPECT_RATIOS


def test_filament_mask_predictor_output_shape():
    """FilamentMaskPredictor output shape must match torchvision default."""
    import torch

    predictor = FilamentMaskPredictor(in_channels=256, dim_reduced=256, num_classes=2)
    # Simulate MaskRCNNHeads output: 14×14 features per proposal.
    x = torch.randn(8, 256, 14, 14)
    out = predictor(x)
    assert out.shape == (8, 2, 56, 56)


def test_filament_mask_predictor_pixel_shuffle_learnable():
    """PixelShuffle stages must contain learned conv layers (not fixed)."""
    import torch

    predictor = FilamentMaskPredictor(in_channels=256, dim_reduced=256, num_classes=2)
    x = torch.randn(2, 256, 14, 14, requires_grad=True)
    out = predictor(x)
    out.sum().backward()
    grad_norms = [p.grad.norm().item() for p in predictor.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_norms)


def test_a5_5_config_use_filament_net_default_off():
    """Default config must keep FilamentNet off (single-variable discipline)."""
    config = A5_5InstanceConfig()
    assert config.use_filament_net is False


def test_a5_5_config_use_filament_net_can_be_enabled():
    config = A5_5InstanceConfig(use_filament_net=True)
    assert config.use_filament_net is True


def test_build_filament_maskrcnn_returns_maskrcnn_instance():
    """build_filament_maskrcnn must return a torchvision MaskRCNN instance."""
    from torchvision.models.detection import MaskRCNN

    config = A5_5InstanceConfig(use_filament_net=True)
    model = build_filament_maskrcnn(config)
    assert isinstance(model, MaskRCNN)
    assert isinstance(model.roi_heads.mask_predictor, FilamentMaskPredictor)


# ---------------------------------------------------------------------------
# Metrics bridge tests.
# ---------------------------------------------------------------------------


def test_resize_mask_to_eval_shape_changes_to_target():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    out = _resize_mask_to_eval(mask, (16, 16))
    assert out.shape == (16, 16)
    assert out.dtype == np.uint8
    assert out.sum() > 0


def test_maskrcnn_predictions_to_instance_masks_filters_low_scores():
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, 2:6, 2:6] = 1
    masks[1, 3:5, 3:5] = 1
    prediction = MaskRCNNPrediction(
        image_id="x",
        scores=np.array([0.9, 0.3], dtype=np.float32),
        boxes=np.zeros((2, 4), dtype=np.float32),
        masks=masks,
        model_grid_hw=(8, 8),
    )
    out = maskrcnn_predictions_to_instance_masks(
        prediction,
        score_threshold=0.5,
        mask_threshold=0.5,
        min_instance_area=1,
        max_instances=10,
        evaluation_resolution=(32, 32),
    )
    assert len(out) == 1


def test_maskrcnn_predictions_to_instance_masks_filters_small_area():
    masks = np.zeros((1, 8, 8), dtype=np.uint8)
    masks[0, 0:1, 0:1] = 1
    prediction = MaskRCNNPrediction(
        image_id="x",
        scores=np.array([0.9], dtype=np.float32),
        boxes=np.zeros((1, 4), dtype=np.float32),
        masks=masks,
        model_grid_hw=(8, 8),
    )
    out = maskrcnn_predictions_to_instance_masks(
        prediction,
        score_threshold=0.5,
        mask_threshold=0.5,
        min_instance_area=4,
        max_instances=10,
        evaluation_resolution=(32, 32),
    )
    assert len(out) == 0  # filtered by area


def test_evaluate_maskrcnn_predictions_handles_empty_gt():
    masks = np.zeros((1, 8, 8), dtype=np.uint8)
    masks[0, 2:6, 2:6] = 1
    prediction = MaskRCNNPrediction(
        image_id="x",
        scores=np.array([0.9], dtype=np.float32),
        boxes=np.zeros((1, 4), dtype=np.float32),
        masks=masks,
        model_grid_hw=(8, 8),
    )
    from filament_contract.a5_strict_pq_contract import StrictPQContract

    contract = StrictPQContract(
        evaluation_resolution=(16, 16),
        probability_threshold=0.5,
        min_instance_area=1,
        connectivity=8,
        max_instances=10,
        match_iou_threshold=0.5,
    )
    metrics = evaluate_maskrcnn_predictions(
        predictions=[prediction],
        ground_truth_by_image={"x": []},
        contract=contract,
    )
    assert len(metrics) == 1
    pq, image_id = metrics[0]
    assert image_id == "x"
    # With no GT and 1 predicted instance: it's a false positive.
    assert pq.tp_count == 0
    assert pq.fp_count == 1
    assert pq.fn_count == 0
    assert pq.pq_like == 0.0


def test_evaluate_maskrcnn_predictions_perfect_match():
    from filament_contract.a5_strict_pq_contract import StrictPQContract

    # Create a single GT mask and a single matching prediction.
    gt = np.zeros((16, 16), dtype=bool)
    gt[4:12, 4:12] = True
    masks = np.zeros((1, 16, 16), dtype=np.uint8)
    masks[0, 4:12, 4:12] = 1
    prediction = MaskRCNNPrediction(
        image_id="y",
        scores=np.array([0.95], dtype=np.float32),
        boxes=np.zeros((1, 4), dtype=np.float32),
        masks=masks,
        model_grid_hw=(16, 16),
    )
    contract = StrictPQContract(
        evaluation_resolution=(16, 16),
        probability_threshold=0.5,
        min_instance_area=1,
        connectivity=8,
        max_instances=10,
        match_iou_threshold=0.5,
    )
    metrics = evaluate_maskrcnn_predictions(
        predictions=[prediction],
        ground_truth_by_image={"y": [gt]},
        contract=contract,
    )
    pq, _ = metrics[0]
    assert pq.tp_count == 1
    assert pq.fp_count == 0
    assert pq.fn_count == 0
    assert pq.pq_like > 0.99  # near-perfect IoU under nearest-neighbor resize