"""A5-4 Phase 1: Feature Pyramid Network decoder for multi-scale filament segmentation.

This module implements the FPN decoder architecture for Phase 1 (WP1) of the A5-4
architectural specialization program. FPN provides multi-scale context through
top-down pathways and lateral connections, hypothesized to better capture both
fine filament details and global structure compared to U-Net++'s nested dense skips.

**Single variable changed from baseline**:
- Decoder: U-Net++ nested skip connections → FPN top-down pathway

**Frozen**:
- Encoder: ResNet34, ImageNet init, 768 input
- Loss: BCE + soft Dice (1:1 weight)
- Training: epochs/LR/optimizer/batch/D4 augmentation
- Postprocessing: PQ contract (threshold=0.50, min_area=256)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FilamentFPN(nn.Module):
    """Feature Pyramid Network decoder for multi-scale solar filament segmentation.

    Architecture:
    - Top-down pathway: P5 → P4 → P3 → P2
    - Lateral connections from ResNet34 encoder stages
    - 256 channels per pyramid level
    - Final prediction from P2 (1/4 scale) upsampled to full resolution
    - Single binary segmentation output head

    The FPN design captures multi-scale context by:
    1. High-level semantic features from P5 (1/32 resolution)
    2. Progressive refinement through P4, P3, P2
    3. Lateral connections preserve fine spatial detail from encoder

    This should improve detection of:
    - Thin filament tendrils (fine detail from P2)
    - Large filament bodies (context from P5)
    - Multi-scale continuity (top-down pathway)
    """

    def __init__(
        self,
        encoder_channels: tuple[int, ...] = (64, 64, 128, 256, 512),
        fpn_channels: int = 256,
        num_classes: int = 1,
    ):
        """Initialize FPN decoder.

        Args:
            encoder_channels: Channel counts from encoder stages 1-5 (ResNet34: 64,64,128,256,512)
            fpn_channels: Channel count for all FPN pyramid levels (default 256)
            num_classes: Output classes (1 for binary segmentation)
        """
        super().__init__()

        self.encoder_channels = encoder_channels
        self.fpn_channels = fpn_channels

        # Lateral connections: reduce encoder channels to FPN channels
        # Stage 5 (bottom) to Stage 2 (top, 1/4 resolution)
        self.lateral5 = nn.Conv2d(encoder_channels[4], fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(encoder_channels[3], fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(encoder_channels[2], fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(encoder_channels[1], fpn_channels, kernel_size=1)

        # Top-down pathway: smooth upsampled features after addition
        # Use 3×3 conv to reduce aliasing from upsampling
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        # Final prediction head from P2
        # P2 is at 1/4 resolution, will be upsampled 4× to full resolution
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through FPN decoder.

        Args:
            encoder_features: List of 5 encoder feature maps [c1, c2, c3, c4, c5]
                c1: Stage 1 (1/2 resolution)  - not used in FPN
                c2: Stage 2 (1/4 resolution)  - P2
                c3: Stage 3 (1/8 resolution)  - P3
                c4: Stage 4 (1/16 resolution) - P4
                c5: Stage 5 (1/32 resolution) - P5

        Returns:
            Logits tensor (B, 1, H, W) at full input resolution
        """
        # Unpack encoder features (ResNet34 stages)
        c1, c2, c3, c4, c5 = encoder_features

        # Build pyramid from top (P5) to bottom (P2)
        # P5: Start with deepest features
        p5 = self.lateral5(c5)

        # P4: Upsample P5 and add lateral connection from C4
        p4 = self._upsample_add(p5, self.lateral4(c4))
        p4 = self.smooth4(p4)

        # P3: Upsample P4 and add lateral connection from C3
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p3 = self.smooth3(p3)

        # P2: Upsample P3 and add lateral connection from C2
        p2 = self._upsample_add(p3, self.lateral2(c2))
        p2 = self.smooth2(p2)

        # Generate segmentation from P2 (1/4 resolution)
        seg_logits = self.segmentation_head(p2)

        # Upsample to full resolution (4×)
        # Input was downsampled to 1/4 by encoder stage 2
        output = F.interpolate(
            seg_logits,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )

        return output

    def _upsample_add(
        self,
        top_feature: torch.Tensor,
        lateral_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Upsample top feature 2× and add to lateral feature.

        Args:
            top_feature: Feature map from higher pyramid level (lower resolution)
            lateral_feature: Feature map from encoder stage (higher resolution)

        Returns:
            Element-wise sum of upsampled top and lateral features
        """
        # Upsample top feature to match lateral feature resolution
        upsampled = F.interpolate(
            top_feature,
            size=lateral_feature.shape[2:],
            mode="nearest",
        )
        # Element-wise addition
        return upsampled + lateral_feature


def build_fpn_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    input_channels: int = 3,
) -> nn.Module:
    """Build complete FPN model with ResNet encoder.

    Args:
        encoder_name: Encoder architecture (default "resnet34")
        encoder_weights: Pretrained weights (default "imagenet")
        input_channels: Input channels (default 3 for RGB)

    Returns:
        Complete model with encoder + FPN decoder
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise RuntimeError(
            "Phase 1 FPN requires segmentation_models_pytorch; "
            "install with: pip install segmentation-models-pytorch"
        ) from e

    # Build encoder
    if encoder_name != "resnet34":
        raise ValueError("Phase 1 requires ResNet34 encoder (frozen from baseline)")

    # Use SMP to get encoder with pretrained weights
    encoder = smp.encoders.get_encoder(
        encoder_name,
        in_channels=input_channels,
        depth=5,
        weights=encoder_weights,
    )

    # Get encoder output channels
    encoder_channels = encoder.out_channels  # (3, 64, 64, 128, 256, 512) for ResNet34

    # Build FPN decoder
    decoder = FilamentFPN(
        encoder_channels=encoder_channels[1:],  # Skip RGB input channels
        fpn_channels=256,
        num_classes=1,
    )

    # Wrap in a complete model
    class FPNSegmentationModel(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            features = self.encoder(x)
            output = self.decoder(features)
            return output

    return FPNSegmentationModel(encoder, decoder)


__all__ = [
    "FilamentFPN",
    "build_fpn_model",
]
