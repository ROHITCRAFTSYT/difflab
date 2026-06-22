"""Shared pytest fixtures: tiny models/schedulers that run fast on CPU."""

from __future__ import annotations

import pytest
import torch

from difflab.config import ModelConfig, SchedulerConfig
from difflab.models import build_scheduler, build_unet


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _tiny_model_config(num_classes: int = 0) -> ModelConfig:
    """A minimal 2-level UNet on 8x8 single-channel images."""
    return ModelConfig(
        sample_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        norm_num_groups=8,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        num_classes=num_classes,
    )


@pytest.fixture
def tiny_model_config():
    return _tiny_model_config()


@pytest.fixture
def tiny_unet():
    return build_unet(_tiny_model_config())


@pytest.fixture
def tiny_cond_unet():
    return build_unet(_tiny_model_config(num_classes=3))


@pytest.fixture
def scheduler_config():
    return SchedulerConfig(num_train_timesteps=50)


@pytest.fixture
def ddpm_scheduler(scheduler_config):
    return build_scheduler(scheduler_config, kind="ddpm")


@pytest.fixture
def ddim_scheduler(scheduler_config):
    return build_scheduler(scheduler_config, kind="ddim")


@pytest.fixture
def ddim_inv_scheduler(scheduler_config):
    """DDIM scheduler configured for inversion (no x0 clipping)."""
    return build_scheduler(scheduler_config, kind="ddim", clip_sample=False, set_alpha_to_one=False)
