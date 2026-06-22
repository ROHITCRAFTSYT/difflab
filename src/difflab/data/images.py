"""Image dataset loading + preprocessing.

Wraps the ``datasets`` library so any image dataset on the Hub (or a local
``imagefolder``) can be turned into a diffusion-ready dataloader yielding
batches of ``{"images": FloatTensor[-1,1], "labels": LongTensor | None}``.

Transforms are implemented with PIL + numpy (no torchvision dependency) so the
package installs cleanly against any PyTorch build, including CPU-only wheels.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from difflab.config import DataConfig


def preprocess_image(
    img: Image.Image,
    resolution: int,
    channels: int,
    center_crop: bool = True,
    random_flip: bool = False,
) -> torch.Tensor:
    """Resize/crop/normalize a PIL image to a CHW float tensor in [-1, 1]."""
    img = img.convert("L") if channels == 1 else img.convert("RGB")

    # Resize so the short side == resolution, then optionally center-crop a square.
    w, h = img.size
    scale = resolution / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    if center_crop:
        w, h = img.size
        left, top = (w - resolution) // 2, (h - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))
    else:
        img = img.resize((resolution, resolution), Image.BICUBIC)

    if random_flip and random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # [0, 1]
    if arr.ndim == 2:
        arr = arr[..., None]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    return tensor * 2.0 - 1.0  # [-1, 1]


def build_image_dataloader(
    cfg: DataConfig,
    channels: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build a ``DataLoader`` over an image dataset described by ``cfg``.

    The dataset is loaded lazily via ``datasets.load_dataset``. If
    ``cfg.label_column`` is set, the batch dict includes integer ``labels``.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, split=cfg.split)
    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))

    image_col = cfg.image_column
    label_col = cfg.label_column

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = torch.stack(
            [
                preprocess_image(
                    item[image_col], cfg.resolution, channels, cfg.center_crop, cfg.random_flip
                )
                for item in batch
            ]
        )
        out = {"images": images}
        if label_col is not None:
            out["labels"] = torch.tensor(
                [int(item[label_col]) for item in batch], dtype=torch.long
            )
        return out

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
    )
