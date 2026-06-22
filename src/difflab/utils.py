"""Shared utilities: reproducibility, device selection, image grids, logging."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_LOGGER_NAME = "difflab"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a configured logger that writes a single concise stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs for reproducible runs.

    Setting ``deterministic`` favours reproducibility over speed by disabling
    cuDNN autotuning. This is cheap on CPU and recommended for submissions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device.

    Order of preference: explicit ``prefer`` -> CUDA -> Apple MPS -> CPU.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    """Convert a batch of images in [-1, 1] (B, C, H, W) to a list of PIL images."""
    images = (images.clamp(-1, 1) + 1) / 2  # -> [0, 1]
    images = (images * 255).round().to(torch.uint8).cpu()
    images = images.permute(0, 2, 3, 1).numpy()  # B, H, W, C
    out = []
    for arr in images:
        if arr.shape[-1] == 1:
            out.append(Image.fromarray(arr.squeeze(-1), mode="L"))
        else:
            out.append(Image.fromarray(arr, mode="RGB"))
    return out


def make_image_grid(images: list[Image.Image], rows: int | None = None, cols: int | None = None) -> Image.Image:
    """Tile a list of PIL images into a single grid image."""
    if not images:
        raise ValueError("make_image_grid received an empty image list.")
    n = len(images)
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    elif rows is None:
        rows = int(np.ceil(n / cols))
    elif cols is None:
        cols = int(np.ceil(n / rows))

    w, h = images[0].size
    mode = images[0].mode
    grid = Image.new(mode, size=(cols * w, rows * h), color=0)
    for idx, img in enumerate(images):
        if img.size != (w, h):
            img = img.resize((w, h))
        grid.paste(img, box=((idx % cols) * w, (idx // cols) * h))
    return grid


def count_parameters(module: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
