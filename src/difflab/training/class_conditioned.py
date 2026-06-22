"""Pillar 2 — train a class-conditioned diffusion model from scratch.

Builds a label-conditioned ``UNet2DModel`` and trains it with the shared
:class:`~difflab.training.trainer.Trainer`. After training you can sample any
class on demand (see :func:`difflab.sampling.ddim_sample` with ``class_labels``).
"""

from __future__ import annotations

from difflab.config import ExperimentConfig
from difflab.data.images import build_image_dataloader
from difflab.models import build_scheduler, build_unet
from difflab.training.trainer import Trainer, TrainResult
from difflab.utils import count_parameters, get_logger, set_seed

logger = get_logger()


def run(cfg: ExperimentConfig) -> TrainResult:
    """Train a class-conditioned model end-to-end from a config."""
    if cfg.model.num_classes <= 0:
        raise ValueError("class_conditioned requires model.num_classes > 0.")
    if cfg.data.label_column is None:
        raise ValueError("class_conditioned requires data.label_column to be set.")

    set_seed(cfg.train.seed)
    model = build_unet(cfg.model)
    noise_scheduler = build_scheduler(cfg.scheduler, kind="ddpm")
    dataloader = build_image_dataloader(
        cfg.data,
        channels=cfg.model.in_channels,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.dataloader_num_workers,
    )
    logger.info(
        "class-conditioned UNet: %s params, %d classes",
        f"{count_parameters(model):,}",
        cfg.model.num_classes,
    )
    return Trainer(model, noise_scheduler, dataloader, cfg).train()
