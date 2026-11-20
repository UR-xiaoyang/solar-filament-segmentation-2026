"""PyTorch model and loss factories for the M2 semantic baseline."""

from __future__ import annotations

import os
from typing import Any

from .m2_config import M2Config


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def require_training_stack() -> tuple[Any, Any]:
    # SMP falls back to its checksum-pinned original URL when the HF hub is offline.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import segmentation_models_pytorch as smp
        import torch
    except ImportError as error:
        raise RuntimeError(
            "M2 training requires the 'train' optional dependencies; "
            "install the CUDA PyTorch build and then `pip install -e '.[train]'`"
        ) from error
    return torch, smp


def input_channel_count(config: M2Config) -> int:
    config.validate()
    return 2 if config.input_feature_mode == "local_contrast_residual" else 1


def build_model(config: M2Config) -> Any:
    """Build the exact architecture named by the frozen configuration."""

    config.validate()
    _, smp = require_training_stack()
    return smp.UnetPlusPlus(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )


def prepare_model_input(image: Any, config: M2Config) -> Any:
    """Adapt configured grayscale/feature stacks to pretrained RGB encoders."""

    torch, _ = require_training_stack()
    expected_channels = input_channel_count(config)
    if image.ndim != 4 or image.shape[1] != expected_channels:
        raise ValueError(
            f"model input must have shape [batch, {expected_channels}, height, width]"
        )
    if expected_channels == 1:
        image = image.repeat(1, 3, 1, 1)
    else:
        # Preserve pretrained ImageNet compatibility while adding one engineered feature:
        # R/G carry the original image, B carries the residual channel.
        image = torch.cat([image[:, :1], image[:, :1], image[:, 1:2]], dim=1)
    if config.pretrained_input_normalization:
        mean = torch.as_tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device)[
            None, :, None, None
        ]
        std = torch.as_tensor(IMAGENET_STD, dtype=image.dtype, device=image.device)[
            None, :, None, None
        ]
        image = (image - mean) / std
    return image


def build_bce_dice_loss(config: M2Config) -> Any:
    """Return BCE-with-logits plus soft Dice, both valid for vote-fraction labels."""

    config.validate()
    torch, _ = require_training_stack()

    class BCEDiceLoss(torch.nn.Module):
        def forward(self, logits: Any, target: Any) -> tuple[Any, dict[str, Any]]:
            if logits.shape != target.shape:
                raise ValueError("logits and target shapes differ")
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            probability = torch.sigmoid(logits)
            axes = tuple(range(1, target.ndim))
            intersection = (probability * target).sum(dim=axes)
            denominator = probability.sum(dim=axes) + target.sum(dim=axes)
            dice = ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
            dice_loss = 1.0 - dice
            total = config.bce_weight * bce + config.dice_weight * dice_loss
            return total, {"bce": bce.detach(), "soft_dice": dice.detach()}

    return BCEDiceLoss()
