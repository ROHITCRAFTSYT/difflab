"""Scheduler factory + forward-noising math."""

from __future__ import annotations

import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from difflab.config import SchedulerConfig
from difflab.models import build_scheduler
from difflab.models.schedulers import matching_ddim


def test_build_scheduler_kinds():
    cfg = SchedulerConfig(num_train_timesteps=123)
    assert isinstance(build_scheduler(cfg, "ddpm"), DDPMScheduler)
    assert isinstance(build_scheduler(cfg, "ddim"), DDIMScheduler)
    assert build_scheduler(cfg, "ddpm").config.num_train_timesteps == 123


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_scheduler(SchedulerConfig(), "nope")


def test_matching_ddim_shares_betas(ddpm_scheduler):
    ddim = matching_ddim(ddpm_scheduler)
    assert ddim.config.num_train_timesteps == ddpm_scheduler.config.num_train_timesteps
    assert torch.allclose(ddim.alphas_cumprod, ddpm_scheduler.alphas_cumprod)


def test_add_noise_preserves_shape_and_scales(ddpm_scheduler):
    x0 = torch.randn(4, 1, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.zeros(4, dtype=torch.long)  # t=0 -> almost no noise
    noisy = ddpm_scheduler.add_noise(x0, noise, t)
    assert noisy.shape == x0.shape
    # At t=0 the signal coefficient is ~1, so noisy stays close to x0.
    assert (noisy - x0).abs().mean() < (noise).abs().mean()


def test_add_noise_matches_closed_form(ddpm_scheduler):
    """The forward process must equal sqrt(acp_t)*x0 + sqrt(1-acp_t)*eps exactly.

    This is the canonical DDPM closed form that the training target relies on; if
    it drifts, the epsilon-prediction objective is silently wrong.
    """
    x0 = torch.randn(3, 1, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 17, 49])
    noisy = ddpm_scheduler.add_noise(x0, noise, t)

    acp = ddpm_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
    expected = acp.sqrt() * x0 + (1 - acp).sqrt() * noise
    assert torch.allclose(noisy, expected, atol=1e-5)
