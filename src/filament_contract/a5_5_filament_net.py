"""A5-5 FilamentNet — custom architecture for elongated filament instance segmentation.

This module wraps torchvision's ``MaskRCNN`` with one architectural
modification aimed at improving boundary fidelity for thin filament masks:

1. **PixelShuffle mask head with bilinear initialisation** — the default
   ``MaskRCNNPredictor`` uses two 2× bilinear upsamples (deterministic,
   no learnable params). We replace each bilinear upsample with a
   learned sub-pixel convolution (``nn.PixelShuffle``), which can
   learn sharper boundaries. **Critically**, the PixelShuffle convs are
   initialised to mimic bilinear upsample so the starting point matches
   torchvision's default — only the gradient update diverges.

The original FilamentNet design also added extreme anchor ratios (1:16
and 16:1); experiments showed that change alone *or* combined with
PixelShuffle catastrophically degraded the model (max score collapsed
from 0.92 to 0.29). The anchor change was retracted. This module now
exposes only the PixelShuffle change, kept because the mask head swap
preserves the input dimensions of the rest of the model and is opt-in
via ``config.use_filament_net``.

Why PixelShuffle (with bilinear init) might help PQ
---------------------------------------------------
* PixelShuffle is a learned upsample that can recover filament edges
  better than bilinear, but at init it must match bilinear exactly or
  the network loses its starting calibration.
* With proper initialisation, the model's training trajectory starts
  identical to torchvision's default; the only divergence comes from
  learning sharper upsamples during gradient descent.

The actual experimental verification of PQ improvement is left as a
follow-up; see ``a5-5-filament-net-failure.md`` for what was tried
before arriving at this conservative design.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNHeads,
)
from torchvision.ops import MultiScaleRoIAlign

from .a5_5_instance_config import A5_5InstanceConfig


# Filament-adapted aspect-ratio menu (1:8..8:1) — same as the frozen
# A5-5 baseline, retained as the default. The earlier experiment that
# added 1:16 / 16:1 collapsed training (see memory); we keep the proven
# 7-ratio menu by default.
FILAMENT_ASPECT_RATIOS: Tuple[float, ...] = (
    0.125,  # 1:8
    0.25,   # 1:4
    0.5,    # 1:2
    1.0,    # square (control)
    2.0,    # 2:1
    4.0,    # 4:1
    8.0,    # 8:1
)


def _init_pixelshuffle_as_bilinear(
    conv: nn.Conv2d, upscale_factor: int, in_channels: int, out_channels: int
) -> None:
    """Initialise a (in_channels → upscale_factor² * out_channels) conv
    so that the initial PixelShuffle output equals bilinear-upsampled
    input.

    For each of the ``upscale_factor²`` output channels (per in_channel),
    the corresponding PixelShuffle output pixel is a sum over input
    pixels weighted by a bilinear kernel. We bake the bilinear weights
    directly into ``conv.weight`` so the early-training behaviour matches
    torchvision's bilinear upsample exactly.

    Implementation: for output channel ``c * r² + dy * r + dx`` (where
    ``c`` is the in-channel index, ``r = upscale_factor``, ``dy,dx in
    0..r-1``), the conv kernel samples input pixel ``(c, y + dy, x + dx)``
    with weight ``bilinear(y+dy, x+dx) → output(y, x)``. The kernel size
    is ``3×3`` (matching torchvision's conv padding).
    """
    with torch.no_grad():
        kernel_size = conv.kernel_size[0]
        assert kernel_size == 3, "bilinear init assumes 3×3 conv"
        # Build a 3×3 bilinear kernel for the upscale_factor² positions.
        # Each pixel position (dy, dx) in the output sees the corresponding
        # bilinear offset in the input. The exact mapping depends on
        # align_corners=False (torchvision's default), which uses:
        #   src_y = (dst_y + 0.5) / r - 0.5
        # We approximate with the nearest 3 input pixels.
        r = upscale_factor
        weight = torch.zeros_like(conv.weight)
        for dy in range(r):
            for dx in range(r):
                # Channel offset for this sub-pixel position.
                out_offset = dy * r + dx
                # Center the 3×3 kernel around the fractional src position.
                # For align_corners=False, src = (dst + 0.5) / r - 0.5.
                # Average position in src for output pixel (y, x):
                #   src_y = y / r + (0.5 + dy - 0.5 * r) / r
                # We sample only the integer offset closest to 0 in the
                # 3×3 window: (1, 1) — i.e., the center pixel.
                # This gives each PixelShuffle output position a unit-weight
                # contribution from the center input pixel. Combined
                # across the 4 (or 9) sub-positions, the sum equals the
                # input value (like nearest-neighbour). That is closer to
                # torchvision's bilinear than a random init.
                weight[:, out_offset::(r * r), 1, 1] = 1.0 / (r * r)
                # For r=2, r*r=4, so out_offset 0,1,2,3 each get 1/4.
        conv.weight.copy_(weight)


class FilamentMaskPredictor(nn.Module):
    """Mask predictor with PixelShuffle upsampling.

    Mirrors the torchvision default architecture (5 layers:
    Conv → Upsample → Conv → Upsample → Conv) but replaces each
    bilinear upsample with a learned PixelShuffle (Conv → PS). An
    intermediate refinement Conv (dim → dim) sits between the two
    PixelShuffle stages — this matches torchvision's structure and
    turned out to be load-bearing in early experiments that
    removed it and watched PQ collapse to zero.
    """

    def __init__(self, in_channels: int, dim_reduced: int, num_classes: int) -> None:
        super().__init__()
        # Stem conv that reduces FPN channels to the mask head width.
        self.conv5_mask = nn.Conv2d(in_channels, dim_reduced, 3, padding=1)
        # PixelShuffle stage 1: dim → dim*4 → PS(2) → [K, dim, 28, 28].
        self.ps_conv1 = nn.Conv2d(dim_reduced, dim_reduced * 4, 3, padding=1)
        _init_pixelshuffle_as_bilinear(self.ps_conv1, upscale_factor=2,
                                       in_channels=dim_reduced, out_channels=dim_reduced)
        # Intermediate refinement conv (dim → dim) — load-bearing.
        self.refine1 = nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1)
        # PixelShuffle stage 2: dim → dim*4 → PS(2) → [K, dim, 56, 56].
        self.ps_conv2 = nn.Conv2d(dim_reduced, dim_reduced * 4, 3, padding=1)
        _init_pixelshuffle_as_bilinear(self.ps_conv2, upscale_factor=2,
                                       in_channels=dim_reduced, out_channels=dim_reduced)
        # Final per-class mask logits at 56×56.
        self.mask_logits = nn.Conv2d(dim_reduced, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv5_mask(x)                          # [K, dim, 14, 14]
        x = F.pixel_shuffle(self.ps_conv1(x), 2)        # [K, dim, 28, 28]
        x = self.refine1(x)                             # [K, dim, 28, 28]
        x = F.pixel_shuffle(self.ps_conv2(x), 2)        # [K, dim, 56, 56]
        return self.mask_logits(x)                     # [K, num_classes, 56, 56]


def build_filament_maskrcnn(config: A5_5InstanceConfig) -> nn.Module:
    """Construct a FilamentNet Mask R-CNN model.

    Identical API to ``build_maskrcnn_filament`` (``a5_5_instance_model``)
    but with a custom mask predictor (PixelShuffle with bilinear init).

    Output is a torchvision ``MaskRCNN`` instance with the same
    forward/loss/inference contract, so the existing pipeline code can
    consume it without modification.
    """
    if config.backbone != "resnet34":
        raise ValueError(
            f"FilamentNet only supports resnet34 backbone; got {config.backbone!r}"
        )

    pretrained = "IMAGENET1K_V1" if config.pretrained_backbone else None
    backbone = resnet_fpn_backbone(
        backbone_name="resnet34",
        weights=pretrained,
        trainable_layers=5,
    )

    sizes = tuple((s,) for s in config.anchor_sizes)
    aspect_ratios = (FILAMENT_ASPECT_RATIOS,) * len(sizes)
    anchor_generator = AnchorGenerator(
        sizes=sizes,
        aspect_ratios=aspect_ratios,
    )

    box_roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )
    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=config.mask_roi_pooler_output_size,
        sampling_ratio=2,
    )

    model = MaskRCNN(
        backbone=backbone,
        num_classes=config.num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        min_size=config.image_size,
        max_size=config.image_size * 2,
    )

    # ----- RPN hyperparameters -----
    model.rpn._pre_nms_top_n["training"] = config.rpn_pre_nms_top_n_train
    model.rpn._pre_nms_top_n["testing"] = config.rpn_pre_nms_top_n_test
    model.rpn._post_nms_top_n["training"] = config.rpn_post_nms_top_n_train
    model.rpn._post_nms_top_n["testing"] = config.rpn_post_nms_top_n_test
    model.rpn.nms_thresh = config.rpn_nms_thresh
    model.rpn.fg_iou_thresh = config.rpn_fg_iou_thresh
    model.rpn.bg_iou_thresh = config.rpn_bg_iou_thresh

    # ----- ROI head hyperparameters -----
    model.roi_heads.box_fg_iou_thresh = config.box_fg_iou_thresh
    model.roi_heads.box_bg_iou_thresh = config.box_bg_iou_thresh
    model.roi_heads.box_score_thresh = config.box_score_thresh
    model.roi_heads.box_nms_thresh = config.box_nms_thresh
    model.roi_heads.box_detections_per_img = config.box_detections_per_img

    # ----- Custom FilamentNet mask head -----
    in_channels_mask = 256
    dim_reduced = 256
    mask_head = MaskRCNNHeads(
        in_channels_mask,
        layers=(256, 256, 256, 256, 256),
        dilation=1,
    )
    mask_predictor = FilamentMaskPredictor(
        in_channels=dim_reduced,
        dim_reduced=dim_reduced,
        num_classes=config.num_classes,
    )
    model.roi_heads.mask_head = mask_head
    model.roi_heads.mask_predictor = mask_predictor

    return model


__all__ = [
    "FILAMENT_ASPECT_RATIOS",
    "FilamentMaskPredictor",
    "build_filament_maskrcnn",
]