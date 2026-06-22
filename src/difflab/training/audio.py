"""Pillar 4 — train an audio diffusion model on Mel spectrograms.

Builds an unconditional image diffusion UNet over single-channel Mel-spectrogram
images (see :mod:`difflab.data.audio`) and trains it with the shared
:class:`~difflab.training.trainer.Trainer`. Sampling produces spectrogram images
that :class:`~difflab.data.audio.MelConverter` turns back into audible waveforms.
"""

from __future__ import annotations

from difflab.config import ExperimentConfig
from difflab.data.audio import build_audio_dataloader
from difflab.models import build_scheduler, build_unet
from difflab.training.trainer import Trainer, TrainResult
from difflab.utils import count_parameters, get_logger, set_seed

logger = get_logger()


def run(cfg: ExperimentConfig) -> TrainResult:
    """Train an audio (Mel-spectrogram) diffusion model from a config."""
    if cfg.model.in_channels != 1 or cfg.model.out_channels != 1:
        raise ValueError("Audio mel images are single-channel; set in/out_channels = 1.")

    set_seed(cfg.train.seed)
    model = build_unet(cfg.model)
    noise_scheduler = build_scheduler(cfg.scheduler, kind="ddpm")
    dataloader = build_audio_dataloader(
        cfg.data,
        sample_size=cfg.model.sample_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.dataloader_num_workers,
    )
    logger.info(
        "audio diffusion UNet: %s params, mel %dx%d @ %d Hz",
        f"{count_parameters(model):,}",
        cfg.model.sample_size,
        cfg.model.sample_size,
        cfg.data.sample_rate,
    )
    return Trainer(model, noise_scheduler, dataloader, cfg).train()
