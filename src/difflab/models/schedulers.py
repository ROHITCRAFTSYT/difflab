"""Noise scheduler factory.

A thin wrapper over ``diffusers`` schedulers so training and sampling share one
source of truth for the noising process. ``DDPMScheduler`` is used for the
training objective; ``DDIMScheduler`` (same betas) is used for fast,
deterministic sampling and for inversion.
"""

from __future__ import annotations

from diffusers import DDIMScheduler, DDPMScheduler

from difflab.config import SchedulerConfig

_KIND = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler}


def build_scheduler(cfg: SchedulerConfig, kind: str = "ddpm", **overrides):
    """Build a scheduler of the requested ``kind`` ("ddpm" or "ddim").

    Both kinds are constructed from identical betas so a model trained with the
    DDPM scheduler can be sampled with the DDIM scheduler without retraining.
    Extra ``overrides`` are forwarded to the scheduler constructor — most useful
    for DDIM inversion, which needs ``clip_sample=False`` so the deterministic
    forward/backward steps remain exact inverses.
    """
    kind = kind.lower()
    if kind not in _KIND:
        raise ValueError(f"Unknown scheduler kind {kind!r}; expected one of {list(_KIND)}.")

    common = {
        "num_train_timesteps": cfg.num_train_timesteps,
        "beta_schedule": cfg.beta_schedule,
        "beta_start": cfg.beta_start,
        "beta_end": cfg.beta_end,
        "prediction_type": cfg.prediction_type,
    }
    common.update(overrides)
    scheduler = _KIND[kind](**common)
    return scheduler


def matching_ddim(ddpm: DDPMScheduler) -> DDIMScheduler:
    """Return a DDIM scheduler sharing an existing DDPM scheduler's config.

    Useful for fast sampling from a model that was trained against ``ddpm``.
    """
    return DDIMScheduler.from_config(ddpm.config)
