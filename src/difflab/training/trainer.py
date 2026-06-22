"""The shared diffusion training loop.

A single :class:`Trainer` implements the DDPM training objective used by every
image pillar (fine-tuning, class-conditioned, audio). It handles:

* mixed precision + gradient accumulation via ``accelerate``
* a cosine LR schedule with warmup
* optional EMA weights
* periodic checkpointing and sample-grid generation
* TensorBoard logging

The loop is deliberately small and readable; pillar modules just build the
model/scheduler/dataloader and hand them to the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup

from difflab.config import ExperimentConfig
from difflab.models.schedulers import matching_ddim
from difflab.sampling.ddim import ddim_sample
from difflab.training.ema import EMA
from difflab.utils import ensure_dir, get_logger, make_image_grid, tensor_to_pil

logger = get_logger()


@dataclass
class TrainResult:
    """Summary returned by :meth:`Trainer.train`."""

    output_dir: str
    global_step: int
    final_loss: float
    checkpoint_dir: str


class Trainer:
    """DDPM trainer for a ``UNet2DModel`` + ``DDPMScheduler``."""

    def __init__(self, model, noise_scheduler, dataloader, cfg: ExperimentConfig):
        self.cfg = cfg
        self.noise_scheduler = noise_scheduler
        self.dataloader = dataloader

        self.accelerator = Accelerator(
            mixed_precision=cfg.train.mixed_precision,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=str(Path(cfg.train.output_dir) / "logs"),
        )

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

        steps_per_epoch = max(1, len(dataloader) // cfg.train.gradient_accumulation_steps)
        total_steps = cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(cfg.train.lr_warmup_steps, total_steps),
            num_training_steps=total_steps,
        )

        self.model, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, self.optimizer, dataloader, self.lr_scheduler
        )

        self.ema = EMA(self.accelerator.unwrap_model(self.model), cfg.train.ema_decay) if cfg.train.use_ema else None
        self.global_step = 0
        self._total_steps = total_steps

        if self.accelerator.is_main_process:
            ensure_dir(cfg.train.output_dir)
            self.accelerator.init_trackers(cfg.name)

    @property
    def is_class_conditioned(self) -> bool:
        return self.cfg.model.num_classes > 0

    def _training_step(self, batch) -> torch.Tensor:
        clean = batch["images"]
        noise = torch.randn_like(clean)
        bsz = clean.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean.device
        ).long()
        noisy = self.noise_scheduler.add_noise(clean, noise, timesteps)

        if self.is_class_conditioned:
            pred = self.model(noisy, timesteps, class_labels=batch["labels"]).sample
        else:
            pred = self.model(noisy, timesteps).sample

        # Regression target depends on the scheduler's prediction_type.
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(clean, noise, timesteps)
        else:  # "sample"
            target = clean
        return F.mse_loss(pred, target)

    def train(self) -> TrainResult:
        cfg = self.cfg.train
        last_loss = float("nan")
        stop = False

        for epoch in range(cfg.epochs):
            self.model.train()
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    loss = self._training_step(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.ema is not None:
                        self.ema.update(self.accelerator.unwrap_model(self.model))
                    last_loss = loss.detach().item()
                    self.accelerator.log(
                        {"loss": last_loss, "lr": self.lr_scheduler.get_last_lr()[0]},
                        step=self.global_step,
                    )
                    if cfg.max_steps and self.global_step >= cfg.max_steps:
                        stop = True
                        break

            if self.accelerator.is_main_process:
                logger.info("epoch %d | step %d | loss %.4f", epoch, self.global_step, last_loss)
                if (epoch + 1) % cfg.sample_every_epochs == 0 or stop:
                    self._save_samples(epoch)
                if (epoch + 1) % cfg.save_every_epochs == 0:
                    self.save_checkpoint(Path(cfg.output_dir) / f"checkpoint-epoch{epoch}")
            if stop:
                break

        ckpt = Path(cfg.output_dir) / "final"
        if self.accelerator.is_main_process:
            self.save_checkpoint(ckpt)
        self.accelerator.end_training()
        return TrainResult(
            output_dir=cfg.output_dir,
            global_step=self.global_step,
            final_loss=last_loss,
            checkpoint_dir=str(ckpt),
        )

    @torch.no_grad()
    def _sampling_model(self):
        return self.ema.module if self.ema is not None else self.accelerator.unwrap_model(self.model)

    @torch.no_grad()
    def _save_samples(self, epoch: int) -> None:
        cfg = self.cfg.train
        model = self._sampling_model()
        ddim = matching_ddim(self.noise_scheduler)
        labels = None
        if self.is_class_conditioned:
            n = self.cfg.model.num_classes
            labels = torch.arange(cfg.num_eval_samples, device=self.accelerator.device) % n
        images = ddim_sample(
            model,
            ddim,
            num_samples=cfg.num_eval_samples,
            num_inference_steps=cfg.num_inference_steps,
            class_labels=labels,
            device=self.accelerator.device,
        )
        grid = make_image_grid(tensor_to_pil(images))
        sample_dir = ensure_dir(Path(cfg.output_dir) / "samples")
        grid.save(sample_dir / f"epoch_{epoch:04d}.png")
        logger.info("saved sample grid for epoch %d", epoch)

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist the (EMA, if enabled) UNet in ``diffusers`` format."""
        path = ensure_dir(path)
        model = self._sampling_model()
        model.save_pretrained(path)
        logger.info("saved checkpoint -> %s", path)
