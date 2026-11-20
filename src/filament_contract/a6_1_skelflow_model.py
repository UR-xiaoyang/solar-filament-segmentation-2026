"""
A6-1 SkelFlow Model: Skeleton-guided flow field instance segmentation.

Multi-task network that predicts:
1. Skeleton probability map
2. Flow field (2D vectors pointing toward skeleton)
3. Width profile (filament width at each point)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class SkeletonHead(nn.Module):
    """Prediction head for skeleton detection."""

    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature tensor

        Returns:
            skeleton_logits: [B, 1, H, W]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


class FlowFieldHead(nn.Module):
    """Prediction head for flow field (2D vectors)."""

    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, 2, 1)  # 2 channels for (dx, dy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature tensor

        Returns:
            flow_vectors: [B, 2, H, W] normalized flow vectors
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        flow = self.conv3(x)

        # L2 normalize to unit vectors
        flow_norm = torch.sqrt(flow[:, 0:1]**2 + flow[:, 1:2]**2 + 1e-6)
        flow_normalized = flow / flow_norm

        return flow_normalized


class WidthHead(nn.Module):
    """Prediction head for width profile."""

    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature tensor

        Returns:
            width_map: [B, 1, H, W] width in pixels (ReLU to ensure positive)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        width = F.relu(self.conv3(x))  # Width must be positive
        return width


class DecoderBlock(nn.Module):
    """U-Net style decoder block with skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, C1, H, W] upsampled features
            skip: [B, C2, H, W] skip connection from encoder (optional)

        Returns:
            [B, out_channels, H, W]
        """
        # Upsample if needed
        if skip is not None and x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        # Concatenate skip connection
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class SkelFlowNet(nn.Module):
    """
    SkelFlow architecture: ResNet34 encoder + U-Net decoder + 3 prediction heads.

    Architecture:
        - Encoder: ResNet34 (pretrained)
        - Decoder: U-Net style with skip connections
        - Heads: Skeleton, Flow Field, Width
    """

    def __init__(self, pretrained: bool = True, encoder_channels: list[int] | None = None):
        super().__init__()

        # Default ResNet34 channel configuration
        if encoder_channels is None:
            encoder_channels = [64, 64, 128, 256, 512]

        # Encoder: ResNet34
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
            resnet = resnet34(weights=weights)
        else:
            resnet = resnet34(weights=None)

        # Extract encoder layers
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 channels
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Decoder blocks (reverse pyramid)
        self.decoder4 = DecoderBlock(512, 256, 256)  # From layer4 + layer3
        self.decoder3 = DecoderBlock(256, 128, 128)  # From decoder4 + layer2
        self.decoder2 = DecoderBlock(128, 64, 64)    # From decoder3 + layer1
        self.decoder1 = DecoderBlock(64, 64, 64)     # From decoder2 + conv1

        # Final upsampling to original resolution
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # Prediction heads
        self.skeleton_head = SkeletonHead(64, mid_channels=64)
        self.flow_head = FlowFieldHead(64, mid_channels=64)
        self.width_head = WidthHead(64, mid_channels=64)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] input image (C=1 for grayscale, C=3 for RGB)

        Returns:
            dict with keys:
                - skeleton_logits: [B, 1, H, W]
                - flow_field: [B, 2, H, W] normalized vectors
                - width_map: [B, 1, H, W] positive values
        """
        input_size = x.shape[-2:]

        # Encoder forward pass
        c1 = self.conv1(x)           # [B, 64, H/2, W/2]
        c1_pool = self.maxpool(c1)   # [B, 64, H/4, W/4]

        c2 = self.layer1(c1_pool)    # [B, 64, H/4, W/4]
        c3 = self.layer2(c2)         # [B, 128, H/8, W/8]
        c4 = self.layer3(c3)         # [B, 256, H/16, W/16]
        c5 = self.layer4(c4)         # [B, 512, H/32, W/32]

        # Decoder forward pass with skip connections
        d4 = self.decoder4(c5, c4)   # [B, 256, H/16, W/16]
        d3 = self.decoder3(d4, c3)   # [B, 128, H/8, W/8]
        d2 = self.decoder2(d3, c2)   # [B, 64, H/4, W/4]
        d1 = self.decoder1(d2, c1)   # [B, 64, H/2, W/2]

        # Final features
        features = self.upsample(d1)  # [B, 64, H, W]

        # Ensure output size matches input
        if features.shape[-2:] != input_size:
            features = F.interpolate(features, size=input_size, mode='bilinear', align_corners=False)

        # Multi-task predictions
        skeleton_logits = self.skeleton_head(features)
        flow_field = self.flow_head(features)
        width_map = self.width_head(features)

        return {
            'skeleton_logits': skeleton_logits,
            'flow_field': flow_field,
            'width_map': width_map,
        }


def build_skelflow_model(pretrained: bool = True) -> SkelFlowNet:
    """
    Factory function to build SkelFlowNet model.

    Args:
        pretrained: Whether to use ImageNet pretrained ResNet34 backbone

    Returns:
        SkelFlowNet model
    """
    return SkelFlowNet(pretrained=pretrained)


if __name__ == "__main__":
    # Test model forward pass
    model = build_skelflow_model(pretrained=False)
    model.eval()

    # Test with dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512)

    with torch.no_grad():
        outputs = model(x)

    print("Model test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Skeleton logits: {outputs['skeleton_logits'].shape}")
    print(f"  Flow field: {outputs['flow_field'].shape}")
    print(f"  Width map: {outputs['width_map'].shape}")

    # Check flow field is normalized
    flow_norm = torch.sqrt(outputs['flow_field'][:, 0]**2 + outputs['flow_field'][:, 1]**2)
    print(f"  Flow field norm (should be ~1.0): mean={flow_norm.mean():.3f}, std={flow_norm.std():.3f}")

    print("\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
