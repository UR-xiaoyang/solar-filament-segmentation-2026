"""
clDice Loss Implementation for Topology-Preserving Segmentation

Reference: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
for Tubular Structure Segmentation", CVPR 2021
GitHub: jocpae/clDice

This module implements centerline Dice (clDice) loss for training neural networks
on tubular/filamentary structures where connectivity preservation is critical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """
    Soft morphological erosion via min pooling.

    Args:
        img: (B, 1, H, W) probability map [0, 1]

    Returns:
        Eroded image (B, 1, H, W)
    """
    # Erosion = -max_pool(-img)
    return 1 - F.max_pool2d(1 - img, kernel_size=3, stride=1, padding=1)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """
    Soft morphological dilation via max pooling.

    Args:
        img: (B, 1, H, W) probability map [0, 1]

    Returns:
        Dilated image (B, 1, H, W)
    """
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """
    Soft morphological opening (erosion followed by dilation).
    Removes small noise while preserving structure shapes.

    Args:
        img: (B, 1, H, W) probability map [0, 1]

    Returns:
        Opened image (B, 1, H, W)
    """
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """
    Differentiable skeletonization via iterative morphological thinning.

    Accumulates residuals from successive erosions to extract the skeleton.
    Higher iterations produce thinner skeletons but may break connectivity
    on faint structures.

    Args:
        img: (B, 1, H, W) probability map [0, 1]
        iterations: Number of erosion iterations (10-15 typical)

    Returns:
        Soft skeleton probability map (B, 1, H, W)
    """
    # Accumulate residuals: skeleton = sum(img_i - erode(img_i))
    skeleton = torch.zeros_like(img)

    for _ in range(iterations):
        eroded = soft_erode(img)
        residual = img - eroded
        skeleton = skeleton + residual
        img = eroded

    return torch.clamp(skeleton, 0, 1)


def soft_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice coefficient for binary segmentation.

    Args:
        pred: (B, 1, H, W) predicted probabilities [0, 1]
        target: (B, 1, H, W) ground truth [0, 1]
        smooth: Smoothing constant to avoid division by zero

    Returns:
        Dice score (scalar tensor)
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def soft_cldice_score(pred: torch.Tensor, target: torch.Tensor,
                      iterations: int = 10, smooth: float = 1.0) -> torch.Tensor:
    """
    Centerline Dice score for topology-aware segmentation.

    Computes Dice on skeletonized predictions and targets, measuring
    connectivity preservation. Formula from paper Eq. 3:
    clDice = (Tprec + Tsens) / 2
    where Tprec = <skel(pred) ∩ target> and Tsens = <pred ∩ skel(target)>

    Args:
        pred: (B, 1, H, W) predicted probabilities [0, 1]
        target: (B, 1, H, W) ground truth [0, 1]
        iterations: Skeletonization iterations
        smooth: Smoothing constant

    Returns:
        clDice score (scalar tensor)
    """
    # Skeletonize both pred and target
    skel_pred = soft_skeletonize(pred, iterations=iterations)
    skel_target = soft_skeletonize(target, iterations=iterations)

    # Topology precision: how much of predicted skeleton overlaps with target
    tprec_num = (skel_pred * target).sum()
    tprec_den = skel_pred.sum()
    tprec = (tprec_num + smooth) / (tprec_den + smooth)

    # Topology sensitivity: how much of target skeleton is recovered
    tsens_num = (pred * skel_target).sum()
    tsens_den = skel_target.sum()
    tsens = (tsens_num + smooth) / (tsens_den + smooth)

    # clDice is average of precision and sensitivity
    cldice = (tprec + tsens) / 2.0
    return cldice


class ClDiceLoss(nn.Module):
    """
    Centerline Dice Loss combining standard Dice and topology-aware clDice.

    Loss = alpha * (1 - Dice) + (1 - alpha) * (1 - clDice)

    Args:
        alpha: Weight between Dice (alpha) and clDice (1-alpha). Default 0.5.
        iterations: Skeletonization iterations. Default 10 for 768px images.
        smooth: Numerical stability constant. Default 1.0.
    """

    def __init__(self, alpha: float = 0.5, iterations: int = 10, smooth: float = 1.0):
        super().__init__()

        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        if smooth <= 0:
            raise ValueError(f"smooth must be > 0, got {smooth}")

        self.alpha = alpha
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute clDice loss.

        Args:
            logits: (B, 1, H, W) raw model outputs
            target: (B, 1, H, W) ground truth [0, 1]

        Returns:
            (loss, metrics_dict)
            loss: Scalar loss tensor with gradient
            metrics_dict: Detached metrics for logging
        """
        # Validate inputs
        if logits.shape != target.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs target {target.shape}")
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"Expected (B, 1, H, W), got {logits.shape}")

        # Apply sigmoid to get probabilities (keep gradient)
        pred = torch.sigmoid(logits)

        # Compute Dice and clDice scores (keep gradient)
        dice = soft_dice_score(pred, target, smooth=self.smooth)
        cldice = soft_cldice_score(pred, target, iterations=self.iterations, smooth=self.smooth)

        # Combine losses (keep gradient on total_loss)
        dice_loss = 1.0 - dice
        cldice_loss = 1.0 - cldice
        total_loss = self.alpha * dice_loss + (1.0 - self.alpha) * cldice_loss

        # Prepare metrics for logging (detach for metrics only)
        with torch.no_grad():
            metrics = {
                'soft_dice': dice.item(),
                'cldice': cldice.item(),
                'dice_loss': dice_loss.item(),
                'cldice_loss': cldice_loss.item(),
                'total_cldice_loss': total_loss.item()
            }

        return total_loss, metrics


class CombinedBCEDiceClDiceLoss(nn.Module):
    """
    Combined loss: BCE + Dice + clDice for comprehensive segmentation training.

    - BCE: Calibrates probabilities, handles class imbalance
    - Dice: Optimizes region overlap
    - clDice: Preserves topology and connectivity

    Loss = w_bce * BCE + w_dice * (1 - Dice) + w_cldice * (1 - clDice)

    Args:
        bce_weight: Weight for BCE loss. Default 0.33.
        dice_weight: Weight for Dice loss. Default 0.33.
        cldice_weight: Weight for clDice loss. Default 0.34.
        cldice_iterations: Skeletonization iterations. Default 10.
        smooth: Numerical stability constant. Default 1.0.
    """

    def __init__(self,
                 bce_weight: float = 0.33,
                 dice_weight: float = 0.33,
                 cldice_weight: float = 0.34,
                 cldice_iterations: int = 10,
                 smooth: float = 1.0):
        super().__init__()

        # Validate weights sum approximately to 1
        total = bce_weight + dice_weight + cldice_weight
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Weights should sum to ~1.0, got {total}")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.cldice_iterations = cldice_iterations
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined BCE + Dice + clDice loss.

        Args:
            logits: (B, 1, H, W) raw model outputs
            target: (B, 1, H, W) ground truth [0, 1]

        Returns:
            (loss, metrics_dict)
        """
        # Validate inputs
        if logits.shape != target.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs target {target.shape}")

        # Get probabilities for Dice/clDice
        pred = torch.sigmoid(logits)

        # 1. BCE loss (with logits for numerical stability)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')

        # 2. Dice loss
        dice_score = soft_dice_score(pred, target, smooth=self.smooth)
        dice_loss = 1.0 - dice_score

        # 3. clDice loss
        cldice_score = soft_cldice_score(pred, target,
                                         iterations=self.cldice_iterations,
                                         smooth=self.smooth)
        cldice_loss = 1.0 - cldice_score

        # Combine with weights (keep gradient on total_loss)
        total_loss = (self.bce_weight * bce +
                     self.dice_weight * dice_loss +
                     self.cldice_weight * cldice_loss)

        # Comprehensive metrics (detach in no_grad context)
        with torch.no_grad():
            metrics = {
                'bce': bce.item(),
                'dice': dice_score.item(),
                'dice_loss': dice_loss.item(),
                'cldice': cldice_score.item(),
                'cldice_loss': cldice_loss.item(),
                'total_loss': total_loss.item()
            }

        return total_loss, metrics
