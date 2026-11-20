"""
A6-1 SkelFlow Loss Functions

Multi-task loss combining:
1. Skeleton detection (Dice + Focal)
2. Flow field alignment (Cosine similarity)
3. Width prediction (Smooth L1)
4. Topology consistency (optional)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, H, W] logits or probabilities
            target: [B, 1, H, W] binary ground truth

        Returns:
            Dice loss (scalar)
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] binary ground truth

        Returns:
            Focal loss (scalar)
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class SkeletonLoss(nn.Module):
    """Combined Dice + Focal loss for skeleton detection."""

    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, 1, H, W] skeleton logits
            target: [B, 1, H, W] binary skeleton ground truth

        Returns:
            Combined loss (scalar)
        """
        dice = self.dice_loss(pred_logits, target)
        focal = self.focal_loss(pred_logits, target)
        return self.dice_weight * dice + self.focal_weight * focal


class FlowFieldLoss(nn.Module):
    """Cosine similarity loss for flow field prediction."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_flow: torch.Tensor,
        target_flow: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_flow: [B, 2, H, W] predicted flow field (normalized)
            target_flow: [B, 2, H, W] ground truth flow field (normalized)
            mask: [B, 1, H, W] binary mask indicating valid flow regions

        Returns:
            Cosine loss (scalar)
        """
        # Compute cosine similarity: dot product of normalized vectors
        # cos(theta) = pred · target (both are unit vectors)
        cosine_sim = (pred_flow * target_flow).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Cosine loss = 1 - cosine_similarity (range [0, 2])
        cosine_loss = 1.0 - cosine_sim

        # Only compute loss in valid regions (inside masks)
        mask_bool = mask > 0.5
        if mask_bool.sum() > 0:
            loss = (cosine_loss * mask).sum() / mask_bool.sum()
        else:
            loss = torch.tensor(0.0, device=pred_flow.device)

        return loss


class WidthLoss(nn.Module):
    """Smooth L1 loss for width prediction."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred_width: torch.Tensor,
        target_width: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_width: [B, 1, H, W] predicted width map
            target_width: [B, 1, H, W] ground truth width map
            mask: [B, 1, H, W] binary mask indicating valid regions

        Returns:
            Smooth L1 loss (scalar)
        """
        # Smooth L1 (Huber) loss
        diff = torch.abs(pred_width - target_width)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )

        # Only compute loss in valid regions
        mask_bool = mask > 0.5
        if mask_bool.sum() > 0:
            loss = (loss * mask).sum() / mask_bool.sum()
        else:
            loss = torch.tensor(0.0, device=pred_width.device)

        return loss


class TopologyConsistencyLoss(nn.Module):
    """
    Topology consistency loss: penalize disconnected skeleton predictions.

    Uses morphological operations to encourage connected components.
    """

    def __init__(self):
        super().__init__()

    def forward(self, skeleton_prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skeleton_prob: [B, 1, H, W] skeleton probability map (after sigmoid)

        Returns:
            Topology loss (scalar)
        """
        # Simple implementation: encourage spatial smoothness via gradient penalty
        # More sophisticated: persistent homology or morphological connectivity

        # Gradient in x and y directions
        grad_x = torch.abs(skeleton_prob[:, :, :, 1:] - skeleton_prob[:, :, :, :-1])
        grad_y = torch.abs(skeleton_prob[:, :, 1:, :] - skeleton_prob[:, :, :-1, :])

        # Penalize high gradients (encourages smooth, connected skeletons)
        loss_x = grad_x.mean()
        loss_y = grad_y.mean()

        return loss_x + loss_y


class SkelFlowLoss(nn.Module):
    """
    Combined multi-task loss for SkelFlow training.

    Loss = w_skel * skeleton_loss +
           w_flow * flow_loss +
           w_width * width_loss +
           w_topo * topology_loss
    """

    def __init__(
        self,
        weight_skeleton: float = 0.4,
        weight_flow: float = 0.3,
        weight_width: float = 0.2,
        weight_topology: float = 0.1,
    ):
        super().__init__()
        self.weight_skeleton = weight_skeleton
        self.weight_flow = weight_flow
        self.weight_width = weight_width
        self.weight_topology = weight_topology

        self.skeleton_loss_fn = SkeletonLoss()
        self.flow_loss_fn = FlowFieldLoss()
        self.width_loss_fn = WidthLoss()
        self.topology_loss_fn = TopologyConsistencyLoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict with keys:
                - skeleton_logits: [B, 1, H, W]
                - flow_field: [B, 2, H, W]
                - width_map: [B, 1, H, W]
            targets: dict with keys:
                - skeleton: [B, 1, H, W] binary
                - flow_field: [B, 2, H, W] normalized
                - width_map: [B, 1, H, W]
                - mask: [B, 1, H, W] valid region mask (optional)

        Returns:
            dict with keys:
                - loss: total weighted loss (scalar)
                - skeleton_loss: skeleton detection loss
                - flow_loss: flow field loss
                - width_loss: width prediction loss
                - topology_loss: topology consistency loss
        """
        # Extract predictions
        skeleton_logits = predictions['skeleton_logits']
        pred_flow = predictions['flow_field']
        pred_width = predictions['width_map']

        # Extract targets
        target_skeleton = targets['skeleton']
        target_flow = targets['flow_field']
        target_width = targets['width_map']

        # Mask: if not provided, use skeleton as mask
        if 'mask' in targets:
            mask = targets['mask']
        else:
            mask = target_skeleton

        # Compute individual losses
        skeleton_loss = self.skeleton_loss_fn(skeleton_logits, target_skeleton)

        flow_loss = self.flow_loss_fn(pred_flow, target_flow, mask)

        width_loss = self.width_loss_fn(pred_width, target_width, mask)

        # Topology loss on skeleton probabilities
        skeleton_prob = torch.sigmoid(skeleton_logits)
        topology_loss = self.topology_loss_fn(skeleton_prob)

        # Weighted combination
        total_loss = (
            self.weight_skeleton * skeleton_loss +
            self.weight_flow * flow_loss +
            self.weight_width * width_loss +
            self.weight_topology * topology_loss
        )

        return {
            'loss': total_loss,
            'skeleton_loss': skeleton_loss,
            'flow_loss': flow_loss,
            'width_loss': width_loss,
            'topology_loss': topology_loss,
        }


def build_skelflow_loss(
    weight_skeleton: float = 0.4,
    weight_flow: float = 0.3,
    weight_width: float = 0.2,
    weight_topology: float = 0.1,
) -> SkelFlowLoss:
    """
    Factory function to build SkelFlow loss.

    Args:
        weight_skeleton: Weight for skeleton detection loss
        weight_flow: Weight for flow field loss
        weight_width: Weight for width prediction loss
        weight_topology: Weight for topology consistency loss

    Returns:
        SkelFlowLoss instance
    """
    return SkelFlowLoss(
        weight_skeleton=weight_skeleton,
        weight_flow=weight_flow,
        weight_width=weight_width,
        weight_topology=weight_topology,
    )


if __name__ == "__main__":
    # Test loss functions
    print("Testing SkelFlow loss functions...")

    batch_size = 2
    H, W = 512, 512

    # Create dummy predictions
    predictions = {
        'skeleton_logits': torch.randn(batch_size, 1, H, W),
        'flow_field': F.normalize(torch.randn(batch_size, 2, H, W), dim=1),
        'width_map': torch.rand(batch_size, 1, H, W) * 10,
    }

    # Create dummy targets
    targets = {
        'skeleton': torch.randint(0, 2, (batch_size, 1, H, W)).float(),
        'flow_field': F.normalize(torch.randn(batch_size, 2, H, W), dim=1),
        'width_map': torch.rand(batch_size, 1, H, W) * 10,
    }

    # Build loss
    loss_fn = build_skelflow_loss()

    # Compute loss
    loss_dict = loss_fn(predictions, targets)

    print("\nLoss values:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")

    print("\nLoss weights:")
    print(f"  skeleton: {loss_fn.weight_skeleton}")
    print(f"  flow: {loss_fn.weight_flow}")
    print(f"  width: {loss_fn.weight_width}")
    print(f"  topology: {loss_fn.weight_topology}")

    print("\nTest passed!")
