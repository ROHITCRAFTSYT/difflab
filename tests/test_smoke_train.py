"""End-to-end smoke test of the Trainer on synthetic data (no network, CPU).

Exercises the full training path: noising, the UNet forward/backward, EMA,
LR schedule, checkpointing, and periodic DDIM sampling — in a couple of seconds.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from difflab.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    SchedulerConfig,
    TrainConfig,
)
from difflab.models import build_scheduler, build_unet
from difflab.training.trainer import Trainer


class _SyntheticImages(Dataset):
    def __init__(self, n: int, channels: int, size: int, num_classes: int):
        self.images = torch.randn(n, channels, size, size).clamp(-1, 1)
        self.labels = torch.randint(0, num_classes, (n,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"images": self.images[idx], "labels": self.labels[idx]}


def _smoke_cfg(output_dir, num_classes, use_ema):
    return ExperimentConfig(
        task="class_conditioned" if num_classes else "finetune",
        name="smoke",
        model=ModelConfig(
            sample_size=16,
            in_channels=1,
            out_channels=1,
            layers_per_block=1,
            norm_num_groups=8,
            block_out_channels=(16, 32),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            num_classes=num_classes,
            pretrained=None if num_classes else "dummy",
        ),
        scheduler=SchedulerConfig(num_train_timesteps=50),
        data=DataConfig(),
        train=TrainConfig(
            output_dir=str(output_dir),
            epochs=1,
            max_steps=2,
            batch_size=4,
            lr_warmup_steps=0,
            use_ema=use_ema,
            sample_every_epochs=1,
            save_every_epochs=1,
            num_eval_samples=4,
            num_inference_steps=4,
        ),
    )


def _run(cfg, num_classes):
    model = build_unet(cfg.model)
    scheduler = build_scheduler(cfg.scheduler, "ddpm")
    ds = _SyntheticImages(16, 1, 16, max(1, num_classes))
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, drop_last=True)
    return Trainer(model, scheduler, loader, cfg).train()


def test_class_conditioned_smoke(tmp_path):
    cfg = _smoke_cfg(tmp_path / "cc", num_classes=3, use_ema=True)
    result = _run(cfg, num_classes=3)
    assert result.global_step == 2
    assert (tmp_path / "cc" / "final").exists()
    # A sample grid was written for the (only) epoch.
    assert list((tmp_path / "cc" / "samples").glob("*.png"))


def test_unconditional_smoke(tmp_path):
    cfg = _smoke_cfg(tmp_path / "un", num_classes=0, use_ema=False)
    result = _run(cfg, num_classes=0)
    assert result.global_step == 2
    assert (tmp_path / "un" / "final").exists()
