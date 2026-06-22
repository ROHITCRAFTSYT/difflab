"""Pillar 1 — fine-tune a pretrained diffusion model on a new dataset.

Loads a pretrained ``UNet2DModel`` (the UNet of a published DDPM pipeline) and
continues training it on a target dataset with the shared
:class:`~difflab.training.trainer.Trainer`. This is dramatically cheaper than
training from scratch and is the recommended way to specialise a model to a new
domain (e.g. a particular set of textures, faces, or objects).
"""

from __future__ import annotations

from difflab.config import ExperimentConfig
from difflab.data.images import build_image_dataloader
from difflab.models import build_scheduler
from difflab.models.unet import load_pretrained_unet
from difflab.training.trainer import Trainer, TrainResult
from difflab.utils import count_parameters, get_logger, set_seed

logger = get_logger()


def run(cfg: ExperimentConfig) -> TrainResult:
    """Fine-tune a pretrained model end-to-end from a config."""
    if not cfg.model.pretrained:
        raise ValueError("finetune requires model.pretrained (a Hub model id).")

    set_seed(cfg.train.seed)
    model = load_pretrained_unet(cfg.model.pretrained)

    # Keep the config in sync with the loaded model so sampling uses the right
    # spatial size / channel count regardless of what the YAML guessed.
    cfg.model.sample_size = model.config.sample_size
    cfg.model.in_channels = model.config.in_channels
    cfg.model.out_channels = model.config.out_channels
    cfg.data.resolution = model.config.sample_size

    noise_scheduler = build_scheduler(cfg.scheduler, kind="ddpm")
    dataloader = build_image_dataloader(
        cfg.data,
        channels=cfg.model.in_channels,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.dataloader_num_workers,
    )
    logger.info(
        "fine-tuning %s (%s params) on %s",
        cfg.model.pretrained,
        f"{count_parameters(model):,}",
        cfg.data.dataset,
    )
    return Trainer(model, noise_scheduler, dataloader, cfg).train()
