"""Tests for A5-4 Phase 1 FPN decoder architecture."""

import pytest
import torch


def test_fpn_decoder_imports():
    """FPN module should import without train dependencies."""
    from filament_contract.a5_fpn_decoder import FilamentFPN
    assert FilamentFPN is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch tests require CUDA"
)
def test_fpn_decoder_builds():
    """FPN decoder should build with ResNet34 encoder channels."""
    from filament_contract.a5_fpn_decoder import FilamentFPN

    # ResNet34 encoder channels: (64, 64, 128, 256, 512)
    decoder = FilamentFPN(
        encoder_channels=(64, 64, 128, 256, 512),
        fpn_channels=256,
        num_classes=1,
    )

    assert decoder is not None
    assert decoder.fpn_channels == 256


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch tests require CUDA"
)
def test_fpn_decoder_forward_shape():
    """FPN decoder should output correct shape."""
    from filament_contract.a5_fpn_decoder import FilamentFPN

    decoder = FilamentFPN(
        encoder_channels=(64, 64, 128, 256, 512),
        fpn_channels=256,
        num_classes=1,
    )

    # Create synthetic encoder features at different resolutions
    # Simulating 768×768 input → stages at 1/2, 1/4, 1/8, 1/16, 1/32
    batch_size = 2
    encoder_features = [
        torch.randn(batch_size, 64, 384, 384),   # c1: 1/2
        torch.randn(batch_size, 64, 192, 192),   # c2: 1/4
        torch.randn(batch_size, 128, 96, 96),    # c3: 1/8
        torch.randn(batch_size, 256, 48, 48),    # c4: 1/16
        torch.randn(batch_size, 512, 24, 24),    # c5: 1/32
    ]

    output = decoder(encoder_features)

    # Output should be full resolution (upsampled 4× from P2)
    assert output.shape == (batch_size, 1, 768, 768)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch tests require CUDA"
)
def test_fpn_pyramid_levels_have_correct_channels():
    """Each FPN pyramid level should have fpn_channels."""
    from filament_contract.a5_fpn_decoder import FilamentFPN

    decoder = FilamentFPN(
        encoder_channels=(64, 64, 128, 256, 512),
        fpn_channels=256,
    )

    # Check lateral connection output channels
    assert decoder.lateral5.out_channels == 256
    assert decoder.lateral4.out_channels == 256
    assert decoder.lateral3.out_channels == 256
    assert decoder.lateral2.out_channels == 256


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch tests require CUDA"
)
def test_build_fpn_model_requires_resnet34():
    """build_fpn_model should enforce ResNet34 encoder."""
    pytest.importorskip("segmentation_models_pytorch")

    from filament_contract.a5_fpn_decoder import build_fpn_model

    # ResNet34 should work
    model = build_fpn_model(encoder_name="resnet34")
    assert model is not None

    # Other encoders should fail (frozen baseline requirement)
    with pytest.raises(ValueError, match="Phase 1 requires ResNet34"):
        build_fpn_model(encoder_name="resnet50")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch tests require CUDA"
)
def test_build_fpn_model_forward():
    """Complete FPN model should process 768×768 input."""
    pytest.importorskip("segmentation_models_pytorch")

    from filament_contract.a5_fpn_decoder import build_fpn_model

    model = build_fpn_model(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        input_channels=3,
    )

    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 768, 768)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (batch_size, 1, 768, 768)
    assert output.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch tests require CUDA"
)
def test_fpn_model_supports_single_channel_input():
    """FPN model should support grayscale input for Phase 1."""
    pytest.importorskip("segmentation_models_pytorch")

    from filament_contract.a5_fpn_decoder import build_fpn_model

    # Grayscale input (filament baseline uses grayscale)
    model = build_fpn_model(
        encoder_name="resnet34",
        encoder_weights=None,  # No pretrained weights for 1-channel
        input_channels=1,
    )

    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 768, 768)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (batch_size, 1, 768, 768)


def test_fpn_decoder_parameter_count_reasonable():
    """FPN should have fewer parameters than U-Net++ (simpler decoder)."""
    pytest.importorskip("segmentation_models_pytorch")

    from filament_contract.a5_fpn_decoder import build_fpn_model

    model = build_fpn_model(encoder_name="resnet34", encoder_weights=None)

    total_params = sum(p.numel() for p in model.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    # ResNet34 encoder ~21M params
    # FPN decoder should be < 10M params (simpler than U-Net++)
    assert total_params < 35_000_000
    assert decoder_params < 10_000_000

    print(f"Total params: {total_params:,}")
    print(f"Decoder params: {decoder_params:,}")
