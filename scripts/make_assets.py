"""Generate the figures embedded in the README, written to ``assets/``.

Figures:
  forward_process.png   — a real image progressively noised (the forward process)
  loss_curve.png        — training loss vs step (from the verify_learning TB logs)
  learning_compare.png  — real vs untrained-samples vs trained-samples, side by side
  beta_schedules.png    — the three supported noise schedules

Run after ``scripts/verify_learning.py`` so the loss curve and trained samples
exist:  python scripts/make_assets.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from difflab.config import SchedulerConfig  # noqa: E402
from difflab.data.images import preprocess_image  # noqa: E402
from difflab.models import build_scheduler  # noqa: E402
from difflab.utils import tensor_to_pil  # noqa: E402

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)
RUN = Path("outputs/verify_learning")


def forward_process_strip():
    """Show one Fashion-MNIST image at increasing noise levels."""
    from datasets import load_dataset

    ds = load_dataset("zalando-datasets/fashion_mnist", split="train")
    img = ds[7]["image"]
    x0 = preprocess_image(img, resolution=64, channels=1, center_crop=True)[None]
    sched = build_scheduler(SchedulerConfig(num_train_timesteps=1000), "ddpm")
    steps = [0, 100, 250, 450, 700, 999]
    noise = torch.randn_like(x0)
    fig, axes = plt.subplots(1, len(steps), figsize=(12, 2.4))
    for ax, t in zip(axes, steps, strict=True):
        noisy = sched.add_noise(x0, noise, torch.tensor([t]))
        ax.imshow(np.asarray(tensor_to_pil(noisy)[0]), cmap="gray")
        ax.set_title(f"t = {t}", fontsize=11)
        ax.axis("off")
    fig.suptitle("Forward diffusion: q(xₜ | x₀) adds noise over 1000 steps", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "forward_process.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/forward_process.png")


def beta_schedules():
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ["linear", "scaled_linear", "squaredcos_cap_v2"]:
        sched = build_scheduler(SchedulerConfig(num_train_timesteps=1000, beta_schedule=name), "ddpm")
        ax.plot(sched.alphas_cumprod.numpy(), label=name)
    ax.set_xlabel("timestep t")
    ax.set_ylabel(r"$\bar{\alpha}_t$ (signal retained)")
    ax.set_title("Noise schedules: cumulative alpha vs timestep")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "beta_schedules.png", dpi=110)
    plt.close(fig)
    print("wrote assets/beta_schedules.png")


def loss_curve():
    from tensorboard.backend.event_processing import event_accumulator

    files = sorted(glob.glob(str(RUN / "logs" / "**" / "events*"), recursive=True))
    if not files:
        print("! no TB logs found; skipping loss_curve")
        return
    ea = event_accumulator.EventAccumulator(files[-1])
    ea.Reload()
    loss = ea.Scalars("loss")
    steps = [s.step for s in loss]
    vals = [s.value for s in loss]
    # light smoothing for readability
    sm = np.convolve(vals, np.ones(5) / 5, mode="valid")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, vals, alpha=0.3, color="tab:blue", label="loss")
    ax.plot(steps[4:], sm, color="tab:blue", label="smoothed")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("MSE loss (ε-prediction)")
    ax.set_title(f"Training loss: {vals[0]:.3f} → {min(vals):.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "loss_curve.png", dpi=110)
    plt.close(fig)
    print(f"wrote assets/loss_curve.png (first {vals[0]:.3f}, min {min(vals):.3f})")


def _load(p):
    return np.asarray(Image.open(p).convert("L"))


def learning_compare():
    triplet = [RUN / "untrained.png", RUN / "trained.png", RUN / "real.png"]
    titles = ["Untrained (random init)", "After training", "Real data (target)"]
    if not all(p.exists() for p in triplet):
        print("! missing run images; skipping learning_compare")
        return
    fig, axes = plt.subplots(1, 3, figsize=(9, 7))
    for ax, p, t in zip(axes, triplet, titles, strict=True):
        ax.imshow(_load(p), cmap="gray")
        ax.set_title(t, fontsize=12)
        ax.axis("off")
    fig.suptitle("Class-conditioned model learns MNIST digits (one digit 0-9 per row)", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSETS / "learning_compare.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/learning_compare.png")


if __name__ == "__main__":
    forward_process_strip()
    beta_schedules()
    loss_curve()
    learning_compare()
