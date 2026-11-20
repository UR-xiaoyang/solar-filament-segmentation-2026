"""
Unit tests for clDice loss implementation

Run: python -m pytest tests/test_cldice_loss.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from filament_contract.cldice_loss import (
    soft_erode, soft_dilate, soft_open, soft_skeletonize,
    soft_dice_score, soft_cldice_score,
    ClDiceLoss, CombinedBCEDiceClDiceLoss
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Morphological Operations Tests ====================

def test_soft_erode_shape():
    """Test erosion preserves shape"""
    img = torch.rand(2, 1, 64, 64)
    eroded = soft_erode(img)
    assert eroded.shape == img.shape


def test_soft_dilate_shape():
    """Test dilation preserves shape"""
    img = torch.rand(2, 1, 64, 64)
    dilated = soft_dilate(img)
    assert dilated.shape == img.shape


def test_soft_open_reduces_noise():
    """Test opening removes small noise"""
    # Create image with small noise speckles
    img = torch.zeros(1, 1, 64, 64)
    img[0, 0, 30:35, 30:35] = 1.0  # Large structure
    img[0, 0, 10, 10] = 1.0  # Single pixel noise

    opened = soft_open(img)

    # Large structure should remain, noise should be reduced
    assert opened[0, 0, 32, 32] > 0.5  # Center of large structure
    assert opened[0, 0, 10, 10] < 0.5  # Noise pixel


# ==================== Skeletonization Tests ====================

def test_soft_skeletonize_thins_structure():
    """Test skeletonization produces thinner output"""
    # Create thick line
    img = torch.zeros(1, 1, 64, 64)
    img[0, 0, 30:35, 10:50] = 1.0  # 5-pixel thick horizontal line

    skeleton = soft_skeletonize(img, iterations=10)

    # Skeleton should have fewer or equal pixels (soft ops may not thin perfectly)
    assert skeleton.sum() <= img.sum()
    # But should not be empty
    assert skeleton.sum() > 0


def test_soft_skeletonize_iterations():
    """Test more iterations produce thinner skeletons"""
    img = torch.ones(1, 1, 64, 64) * 0.8

    skel_5 = soft_skeletonize(img, iterations=5)
    skel_15 = soft_skeletonize(img, iterations=15)

    # More iterations should thin more
    assert skel_15.sum() <= skel_5.sum()


def test_soft_skeletonize_empty_input():
    """Test skeletonization on empty input"""
    img = torch.zeros(1, 1, 64, 64)
    skeleton = soft_skeletonize(img)
    assert skeleton.sum() < 1e-6  # Should remain empty


# ==================== Dice Score Tests ====================

def test_soft_dice_perfect_match():
    """Test Dice score is high for perfect match"""
    pred = torch.ones(1, 1, 64, 64)
    target = torch.ones(1, 1, 64, 64)

    dice = soft_dice_score(pred, target)
    assert dice > 0.99


def test_soft_dice_no_overlap():
    """Test Dice score is low for no overlap"""
    pred = torch.zeros(1, 1, 64, 64)
    pred[0, 0, :32, :] = 1.0

    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 32:, :] = 1.0

    dice = soft_dice_score(pred, target)
    assert dice < 0.1


def test_soft_dice_partial_overlap():
    """Test Dice score for partial overlap"""
    pred = torch.zeros(1, 1, 64, 64)
    pred[0, 0, :48, :] = 1.0

    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 16:, :] = 1.0

    dice = soft_dice_score(pred, target)
    assert 0.3 < dice < 0.9


# ==================== clDice Score Tests ====================

def test_soft_cldice_perfect_match():
    """Test clDice is high for perfect match"""
    pred = torch.ones(1, 1, 64, 64) * 0.9
    target = torch.ones(1, 1, 64, 64)

    cldice = soft_cldice_score(pred, target, iterations=5)
    assert cldice > 0.8


def test_soft_cldice_connectivity():
    """Test clDice rewards connectivity"""
    # Connected structure
    connected = torch.zeros(1, 1, 64, 64)
    connected[0, 0, 32, 10:50] = 1.0  # Horizontal line

    # Broken structure (same pixel count)
    broken = torch.zeros(1, 1, 64, 64)
    broken[0, 0, 32, 10:20] = 1.0
    broken[0, 0, 32, 30:40] = 1.0

    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 32, 10:50] = 1.0

    cldice_connected = soft_cldice_score(connected, target)
    cldice_broken = soft_cldice_score(broken, target)

    # Connected should score higher
    assert cldice_connected > cldice_broken


# ==================== ClDiceLoss Tests ====================

def test_cldice_loss_initialization():
    """Test loss can be initialized with valid parameters"""
    loss = ClDiceLoss(alpha=0.5, iterations=10, smooth=1.0)
    assert loss.alpha == 0.5
    assert loss.iterations == 10


def test_cldice_loss_invalid_alpha():
    """Test loss rejects invalid alpha"""
    with pytest.raises(ValueError):
        ClDiceLoss(alpha=1.5)
    with pytest.raises(ValueError):
        ClDiceLoss(alpha=-0.1)


def test_cldice_loss_forward():
    """Test loss forward pass"""
    loss_fn = ClDiceLoss()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss, metrics = loss_fn(logits, target)

    # Check loss is scalar with gradient
    assert loss.shape == ()
    assert loss.requires_grad

    # Check metrics
    assert 'soft_dice' in metrics
    assert 'cldice' in metrics
    assert 0 <= metrics['soft_dice'] <= 1
    assert 0 <= metrics['cldice'] <= 1


def test_cldice_loss_backward():
    """Test loss gradient flow"""
    loss_fn = ClDiceLoss()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss, _ = loss_fn(logits, target)
    loss.backward()

    # Gradients should exist and not be NaN
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_cldice_loss_shape_validation():
    """Test loss validates input shapes"""
    loss_fn = ClDiceLoss()

    # Mismatched shapes
    with pytest.raises(ValueError):
        logits = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 32, 32)
        loss_fn(logits, target)

    # Wrong number of channels
    with pytest.raises(ValueError):
        logits = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        loss_fn(logits, target)


# ==================== CombinedBCEDiceClDiceLoss Tests ====================

def test_combined_loss_initialization():
    """Test combined loss initialization"""
    loss = CombinedBCEDiceClDiceLoss(
        bce_weight=0.33,
        dice_weight=0.33,
        cldice_weight=0.34
    )
    assert loss.bce_weight == 0.33
    assert loss.dice_weight == 0.33
    assert loss.cldice_weight == 0.34


def test_combined_loss_weight_validation():
    """Test combined loss rejects invalid weights"""
    # Weights don't sum to ~1
    with pytest.raises(ValueError):
        CombinedBCEDiceClDiceLoss(
            bce_weight=0.5,
            dice_weight=0.5,
            cldice_weight=0.5
        )


def test_combined_loss_forward():
    """Test combined loss forward pass"""
    loss_fn = CombinedBCEDiceClDiceLoss()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss, metrics = loss_fn(logits, target)

    # Check loss
    assert loss.shape == ()
    assert loss.requires_grad

    # Check all metrics present
    assert 'bce' in metrics
    assert 'dice' in metrics
    assert 'cldice' in metrics
    assert 'total_loss' in metrics


def test_combined_loss_backward():
    """Test combined loss gradient flow"""
    loss_fn = CombinedBCEDiceClDiceLoss()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss, _ = loss_fn(logits, target)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_combined_loss_batch_processing():
    """Test combined loss handles various batch sizes"""
    loss_fn = CombinedBCEDiceClDiceLoss()

    for batch_size in [1, 4, 8]:
        logits = torch.randn(batch_size, 1, 64, 64)
        target = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()

        loss, metrics = loss_fn(logits, target)
        assert loss.shape == ()
        assert not torch.isnan(loss)


# ==================== GPU Tests ====================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cldice_loss_cuda():
    """Test clDice loss on GPU"""
    device = torch.device("cuda")
    loss_fn = ClDiceLoss().to(device)

    logits = torch.randn(2, 1, 64, 64, device=device)
    target = torch.randint(0, 2, (2, 1, 64, 64), device=device).float()

    loss, metrics = loss_fn(logits, target)
    assert loss.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_combined_loss_cuda():
    """Test combined loss on GPU"""
    device = torch.device("cuda")
    loss_fn = CombinedBCEDiceClDiceLoss().to(device)

    logits = torch.randn(2, 1, 64, 64, device=device)
    target = torch.randint(0, 2, (2, 1, 64, 64), device=device).float()

    loss, metrics = loss_fn(logits, target)
    assert loss.device.type == "cuda"


# ==================== Numerical Stability Tests ====================

def test_dice_score_numerical_stability():
    """Test Dice handles near-zero inputs"""
    pred = torch.zeros(1, 1, 64, 64)
    target = torch.zeros(1, 1, 64, 64)

    dice = soft_dice_score(pred, target, smooth=1.0)
    assert not torch.isnan(dice)
    assert not torch.isinf(dice)


def test_cldice_numerical_stability():
    """Test clDice handles near-zero inputs"""
    pred = torch.zeros(1, 1, 64, 64)
    target = torch.zeros(1, 1, 64, 64)

    cldice = soft_cldice_score(pred, target, smooth=1.0)
    assert not torch.isnan(cldice)
    assert not torch.isinf(cldice)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
