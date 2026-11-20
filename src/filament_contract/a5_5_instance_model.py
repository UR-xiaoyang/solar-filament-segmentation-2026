"""A5-5 Mask R-CNN model factory with filament-adapted anchors.

Builds a torchvision Mask R-CNN with:

* a **ResNet34 FPN backbone** (matches Control-768's encoder so the
  single-variable experiment compares only the architecture);
* a custom **AnchorGenerator** with highly elongated aspect ratios
  (1:8 .. 8:1) and filament-sized widths (16 .. 256 px);
* frozen-from-M2 hyperparameters on RPN and ROI heads.

Why not use Detectron2 or MMDetection?

* No new framework dependency. The ``train`` extra in ``pyproject.toml``
  pins ``torch>=2.7`` and ``torchvision>=0.22``, which already provide
  ``MaskRCNN``, ``resnet_fpn_backbone``, ``AnchorGenerator``, and
  ``MultiScaleRoIAlign``.
* The frozen ``torch`` version is incompatible with current Detectron2
  releases without a custom build, which would compromise the
  reproducibility story.
* torchvision is the canonical reference implementation; if we cannot
  outperform it, we have learned something meaningful.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .a5_5_instance_config import A5_5InstanceConfig


# Map short string identifiers to torchvision's pretrained weight enum.
_BACKBONE_WEIGHTS: Dict[str, str] = {
    "imagenet": "IMAGENET1K_V1",
}


def build_maskrcnn_filament(config: A5_5InstanceConfig) -> nn.Module:
    """Construct a Mask R-CNN model configured for filament segmentation.

    The returned model is in **training mode** (``model.train()`` is
    called inside the training loop). For inference, the caller is
    expected to switch to ``model.eval()`` before producing outputs.
    """
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops import MultiScaleRoIAlign

    if config.backbone != "resnet34":
        raise ValueError(
            f"A5-5 only supports resnet34 backbone; got {config.backbone!r}"
        )

    pretrained = (
        _BACKBONE_WEIGHTS.get("imagenet") if config.pretrained_backbone else None
    )

    # The FPN produces 5 feature maps (level 0 stride 4 ... level pool stride 64).
    # We bind one anchor size to each level and replicate the aspect ratios.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet34",
        weights=pretrained,
        trainable_layers=5,  # freeze none (matches M2 default)
    )

    sizes: Tuple[Tuple[int, ...], ...]
    aspect_ratios: Tuple[float, ...]
    if config.anchor_sizes_per_level is not None:
        sizes = tuple((s,) if isinstance(s, int) else tuple(s)
                       for s in config.anchor_sizes_per_level)
    else:
        sizes = tuple((s,) for s in config.anchor_sizes)
    aspect_ratios = (config.aspect_ratios,) * len(sizes)
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
        # Image-level minimum size for inference (training uses any size).
        min_size=config.image_size,
        max_size=config.image_size * 2,
    )

    # ----- RPN hyperparameters -----
    # torchvision 0.20 changed these from dicts to properties. The
    # backing dicts are private (``_pre_nms_top_n`` etc.) and remain
    # mutable, so we write through them.
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

    return model


def maskrcnn_loss_dict_to_total(losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum the dict of Mask R-CNN losses returned during training.

    Mask R-CNN returns five scalars:
    ``loss_classifier``, ``loss_box_reg``, ``loss_mask``,
    ``loss_objectness``, ``loss_rpn_box_reg``.
    """
    if not losses:
        raise ValueError("Mask R-CNN returned an empty loss dict")
    total = torch.tensor(0.0, dtype=torch.float32)
    for name, tensor in losses.items():
        if not torch.isfinite(tensor):
            raise ValueError(f"non-finite Mask R-CNN loss component: {name}={tensor}")
        total = total + tensor
    return total


def summarize_optimizer_state(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """Return a small JSON-safe dict summarizing parameter group state.

    Used by ``a5_5_instance_pipeline`` to write a reproducible metadata
    snapshot. Parameters and momentum tensors are NOT included; only
    scalars and learning rates.
    """
    summary: Dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        summary[f"group_{index}_lr"] = float(group.get("lr", 0.0))
        summary[f"group_{index}_weight_decay"] = float(group.get("weight_decay", 0.0))
    return summary


__all__ = [
    "build_maskrcnn_filament",
    "maskrcnn_loss_dict_to_total",
    "summarize_optimizer_state",
]