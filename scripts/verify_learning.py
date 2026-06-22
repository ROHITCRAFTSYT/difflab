"""Verification: prove the diffusion model actually *learns* (not just runs).

Trains a class-conditioned model on a Fashion-MNIST subset for a real number of
steps, then writes three grids for visual comparison:

    real.png      — real training images (target distribution)
    untrained.png — samples from the randomly-initialised model (should be noise)
    trained.png   — samples after training (should resemble clothing)

It also prints the per-epoch loss so the downward trend is visible. Designed to
run on CPU in a few minutes. Usage:

    python scripts/verify_learning.py --epochs 12 --samples 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from difflab.config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    SchedulerConfig,
    TrainConfig,
)
from difflab.data.images import build_image_dataloader  # noqa: E402
from difflab.models import build_scheduler, build_unet  # noqa: E402
from difflab.sampling import ddim_sample  # noqa: E402
from difflab.training.trainer import Trainer  # noqa: E402
from difflab.utils import make_image_grid, set_seed, tensor_to_pil  # noqa: E402

CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def sample_grid(model, scheduler, n_per_class, device):
    labels = torch.arange(10, device=device).repeat_interleave(n_per_class)
    imgs = ddim_sample(
        model, scheduler, num_samples=len(labels), num_inference_steps=50,
        class_labels=labels, device=device,
    )
    return make_image_grid(tensor_to_pil(imgs), rows=10, cols=n_per_class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="outputs/verify_learning")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    device = torch.device("cpu")

    cfg = ExperimentConfig(
        task="class_conditioned",
        name="verify_learning",
        model=ModelConfig(
            # Small 16x16, 2-level, no-attention UNet keeps CPU cost low while
            # still clearly learning Fashion-MNIST class structure.
            sample_size=16, in_channels=1, out_channels=1, layers_per_block=1,
            norm_num_groups=8, block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            num_classes=10,
        ),
        scheduler=SchedulerConfig(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"),
        data=DataConfig(
            dataset="zalando-datasets/fashion_mnist", split="train",
            image_column="image", label_column="label", resolution=16,
            center_crop=True, random_flip=False, max_samples=args.samples,
        ),
        train=TrainConfig(
            output_dir=str(out), epochs=args.epochs, batch_size=args.batch,
            learning_rate=2e-4, lr_warmup_steps=100,
            # EMA (decay 0.9999) only helps over many thousands of steps; for this
            # short CPU verification it would keep the sampled weights near random
            # init, so we sample from the actual trained weights instead.
            use_ema=False,
            sample_every_epochs=2,  # progressive sample grids under outputs/.../samples
            save_every_epochs=args.epochs, num_eval_samples=10, num_inference_steps=50,
        ),
    )

    loader = build_image_dataloader(cfg.data, channels=1, batch_size=args.batch, num_workers=0)
    print(f"dataset: {len(loader.dataset)} imgs | steps/epoch={len(loader)} | "
          f"total steps≈{len(loader) * args.epochs}")

    # Real-data reference grid.
    real_batch = next(iter(loader))
    real_imgs = []
    by_label = {i: [] for i in range(10)}
    for img, lbl in zip(real_batch["images"], real_batch["labels"], strict=False):
        if len(by_label[int(lbl)]) < 4:
            by_label[int(lbl)].append(img)
    for i in range(10):
        real_imgs.extend(by_label[i][:4] or [real_batch["images"][0]])
    make_image_grid(tensor_to_pil(torch.stack(real_imgs)), rows=10, cols=4).save(out / "real.png")

    sched_ddim = build_scheduler(cfg.scheduler, kind="ddim")

    # Untrained baseline.
    set_seed(123)
    untrained = build_unet(cfg.model).to(device)
    sample_grid(untrained, sched_ddim, 4, device).save(out / "untrained.png")
    print("wrote real.png + untrained.png")

    # Train (reuses the real product Trainer).
    set_seed(0)
    model = build_unet(cfg.model)
    result = Trainer(model, build_scheduler(cfg.scheduler, "ddpm"), loader, cfg).train()
    print("final loss:", round(result.final_loss, 4))

    # Trained samples from the EMA weights.
    from diffusers import UNet2DModel
    trained = UNet2DModel.from_pretrained(result.checkpoint_dir).to(device)
    sample_grid(trained, build_scheduler(cfg.scheduler, "ddim"), 4, device).save(out / "trained.png")
    print("wrote trained.png ->", out / "trained.png")


if __name__ == "__main__":
    main()
