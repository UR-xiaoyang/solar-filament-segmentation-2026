"""A5-5 FilamentKeypoint — single-stage heatmap detector specialized for
elongated 1D filament structures.

This module replaces Mask R-CNN entirely with a custom architecture
designed around the physics of solar filaments:

1. **Centerline heatmap** — a 1-channel sigmoid output predicting the
   probability of each pixel being on a filament centerline. Trained
   with focal loss against GT skeletons (medial axis of GT polygons).
   This replaces Mask R-CNN's box detection step — heatmap peaks
   are filament centers, naturally handling overlaps without NMS
   suppression.

2. **Width regression head** — a parallel 1-channel output predicting
   the half-width of the filament at each pixel. Trained with L1
   loss against the GT distance-transformed widths, **masked by the
   centerline heatmap** so only centerline pixels contribute gradient.

3. **Mask rasterisation** — at inference, decode peaks via non-max
   suppression, then for each peak rasterize a **disc with the
   predicted radius** as the predicted mask. Resize to the
   2048×2048 evaluation resolution for strict PQ.

Why this should beat Mask R-CNN for filaments
-------------------------------------------
* Mask R-CNN's 28×28 mask output is fundamentally too coarse for
  5–20-pixel-wide filaments — the mask can only represent 1–2 mask
  pixels per filament, which when upsampled fails the strict
  ``min_instance_area=256`` filter.
* Mask R-CNN's box NMS aggressively merges co-oriented filament
  proposals, reducing recall.
* Mask R-CNN's box head treats every detection as a filled 2D region,
  which is the wrong prior for 1D structures.

FilamentKeypoint sidesteps all three:

* Heatmap peaks can be arbitrarily close → no NMS suppression.
* Disc rasterisation produces a continuous, valid mask regardless of
  filament width.
* Focal loss on a smooth Gaussian heatmap target gives the model a
  natural inductive bias towards "1 point per filament" rather than
  "filled box".

Memory budget
-------------
A single ResNet34 backbone + 2 small decoder heads. At 512² input
the peak activation memory is dominated by the upsampled heatmap head
(512×512 × 64 channels ≈ 67 MB). Easily fits at batch=8 on 24 GiB.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GT target generation: skeleton + per-pixel width from COCO polygons.
# ---------------------------------------------------------------------------


def compute_centerline_and_width_from_masks(
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute centerline skeleton and per-pixel width from binary masks.

    Parameters
    ----------
    masks : ``[N, H, W]`` uint8/bool array of binary GT masks.

    Returns
    -------
    centerline : ``[H, W]`` float32 in [0, 1] — Gaussian-blurred skeleton
        of all masks combined (peak intensity 1.0 at skeleton pixels).
    width : ``[H, W]`` float32 — half-width of the filament at each pixel
        (= distance from skeleton to nearest mask boundary), in pixels.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    combined_mask = (masks > 0).any(axis=0)
    if not combined_mask.any():
        return np.zeros(masks.shape[1:], dtype=np.float32), \
               np.zeros(masks.shape[1:], dtype=np.float32)

    # Distance transform: distance from each True pixel to the nearest
    # False pixel. At filament interior, this gives half-width.
    distance = distance_transform_edt(combined_mask.astype(bool))
    # Skeleton: medial axis of combined mask. Use skimage if available,
    # else approximate by distance transform thresholding.
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(combined_mask.astype(bool))
    except ImportError:
        # Approximate skeleton as local maxima of distance transform
        from scipy.ndimage import maximum_filter
        local_max = (distance == maximum_filter(distance, size=5)) & (distance > 1.5)
        skeleton = local_max

    skeleton_f = skeleton.astype(np.float32)
    # Apply Gaussian blur so the heatmap target is smooth (mimicking
    # the spatial uncertainty of "where" the centerline is).
    blurred = gaussian_filter(skeleton_f, sigma=2.0)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()

    # Width: at each pixel, the half-width = distance transform value.
    # Clipped to [1, 50] to avoid unrealistic outliers.
    width = np.clip(distance, 1.0, 50.0).astype(np.float32)

    return blurred, width


def rasterize_disc_mask(
    centers: List[Tuple[float, float]],
    radii: List[float],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Rasterise filament detections as discs on a binary canvas.

    Parameters
    ----------
    centers : list of ``(x, y)`` pixel coordinates of filament centres.
    radii   : list of half-widths in pixels (parallel to width head).
    image_shape : ``(H, W)``.

    Returns
    -------
    mask : ``[H, W]`` uint8 binary mask (1 = inside any disc).
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x, y), r in zip(centers, radii):
        x_i = int(round(x))
        y_i = int(round(y))
        r_i = max(1, int(round(r)))
        y_lo = max(0, y_i - r_i)
        y_hi = min(H, y_i + r_i + 1)
        x_lo = max(0, x_i - r_i)
        x_hi = min(W, x_i + r_i + 1)
        for yy in range(y_lo, y_hi):
            dy = yy - y_i
            for xx in range(x_lo, x_hi):
                dx = xx - x_i
                if dy * dy + dx * dx <= r_i * r_i:
                    mask[yy, xx] = 1
    return mask


# ---------------------------------------------------------------------------
# Focal loss for heatmap regression (adapted from CornerNet / CenterNet).
# ---------------------------------------------------------------------------


def sigmoid_focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Focal loss for heatmap target in [0, 1].

    Predicted logits (pre-sigmoid); target in [0, 1]. Returns mean loss.
    """
    pred_sig = torch.sigmoid(pred_logits)
    pt = pred_sig * target + (1.0 - pred_sig) * (1.0 - target)
    modulator = (1.0 - pt).pow(beta)
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction='none'
    )
    pos_weight = alpha * target
    return (pos_weight * modulator * bce).mean()


# ---------------------------------------------------------------------------
# FilamentKeypoint model.
# ---------------------------------------------------------------------------


class _Decoder(nn.Module):
    """Shared 4×→32× upsampling decoder producing ``out_channels`` feature maps at the original image resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # in feature is at stride 32 of the input. Up to original (×32).
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )
        self.proj = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.proj(x)


class FilamentKeypoint(nn.Module):
    """Heatmap-based detector specialised for filaments.

    Backbone: ImageNet-pretrained ResNet34 (first conv adapted for
    grayscale). Produces a stride-32 feature map.

    Two decoder heads:
    * ``heatmap_head`` → 1-channel logits (sigmoid at inference).
    * ``width_head``   → 1-channel positive width (softplus at inference).
    Both decoders upsample back to the original input resolution.

    Forward returns ``(heatmap_logits, width_pred)`` both of shape
    ``[B, 1, H, W]``.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        from torchvision.models import resnet34

        resnet = resnet34(weights="IMAGENET1K_V1")
        # Adapt first conv to grayscale.
        self.stem_conv = nn.Conv2d(
            in_channels, 64, 7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.stem_conv.weight.copy_(
                resnet.conv1.weight.mean(dim=1, keepdim=True)
            )
        self.stem_bn = resnet.bn1
        self.stem_relu = resnet.relu
        self.stem_maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # output: stride 32

        self.heatmap_head = _Decoder(512, 1)
        self.width_head = _Decoder(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        heatmap_logits = self.heatmap_head(x)  # [B, 1, H, W]
        width_pred = self.width_head(x)  # [B, 1, H, W]
        return heatmap_logits, width_pred


# ---------------------------------------------------------------------------
# Peak decoder at inference (CPU-friendly for small batches).
# ---------------------------------------------------------------------------


def decode_filament_peaks(
    heatmap_logits: torch.Tensor,
    width_pred: torch.Tensor,
    min_distance: int = 8,
    score_threshold: float = 0.3,
    max_peaks: int = 256,
) -> List[Tuple[float, float, float, float]]:
    """Decode filament centres and widths from heatmap + width predictions.

    Returns a list of ``(x, y, radius, score)`` tuples.
    """
    from scipy.ndimage import maximum_filter

    heatmap_sig = torch.sigmoid(heatmap_logits[0, 0]).detach().cpu().numpy().astype(np.float32)
    width_map = (
        F.softplus(width_pred[0, 0]).detach().cpu().numpy().astype(np.float32)
    )  # ensure positive + float32 for scipy

    # Local-max NMS
    local_max = maximum_filter(heatmap_sig, size=min_distance)
    peaks_mask = (heatmap_sig == local_max) & (heatmap_sig >= score_threshold)
    ys, xs = np.where(peaks_mask)
    if len(ys) == 0:
        return []
    scores = heatmap_sig[ys, xs]
    radii = width_map[ys, xs]

    # Sort by score desc, cap at max_peaks
    order = np.argsort(-scores)[:max_peaks]
    return [
        (float(xs[i]), float(ys[i]), float(radii[i]), float(scores[i]))
        for i in order
    ]


__all__ = [
    "FilamentKeypoint",
    "compute_centerline_and_width_from_masks",
    "decode_filament_peaks",
    "rasterize_disc_mask",
    "sigmoid_focal_loss",
]