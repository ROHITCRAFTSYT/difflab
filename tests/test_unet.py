"""UNet factory: shapes, conditioning, and validation."""

from __future__ import annotations

import pytest
import torch

from difflab.config import ModelConfig
from difflab.models import build_unet
from difflab.models.unet import is_class_conditioned


def test_unconditional_forward_shape(tiny_unet):
    assert not is_class_conditioned(tiny_unet)
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([5, 10])
    out = tiny_unet(x, t).sample
    assert out.shape == x.shape


def test_class_conditioned_forward(tiny_cond_unet):
    assert is_class_conditioned(tiny_cond_unet)
    x = torch.randn(3, 1, 8, 8)
    t = torch.tensor([1, 2, 3])
    labels = torch.tensor([0, 1, 2])
    out = tiny_cond_unet(x, t, class_labels=labels).sample
    assert out.shape == x.shape


def test_block_length_mismatch_raises():
    bad = ModelConfig(
        block_out_channels=(8, 16, 32),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    with pytest.raises(ValueError):
        build_unet(bad)
