"""Ancestral DDPM sampling for ``UNet2DModel`` backbones."""

from __future__ import annotations

import torch
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

from difflab.models.unet import is_class_conditioned


@torch.no_grad()
def ddpm_sample(
    model,
    scheduler: DDPMScheduler,
    num_samples: int,
    image_size: int | None = None,
    num_inference_steps: int | None = None,
    class_labels: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
) -> torch.Tensor:
    """Draw samples with the standard ancestral DDPM sampler.

    Returns a tensor in [-1, 1] of shape (num_samples, C, H, W).

    For class-conditioned models, pass ``class_labels`` (a LongTensor of length
    ``num_samples``); if omitted, labels are cycled 0..num_classes-1.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()

    channels = model.config.in_channels
    size = image_size or model.config.sample_size
    if num_inference_steps is not None:
        scheduler.set_timesteps(num_inference_steps, device=device)
    else:
        scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)

    if is_class_conditioned(model):
        num_classes = model.config.num_class_embeds
        if class_labels is None:
            class_labels = torch.arange(num_samples, device=device) % num_classes
        class_labels = class_labels.to(device=device, dtype=torch.long)

    sample = torch.randn(
        (num_samples, channels, size, size), generator=generator, device=device
    )

    iterator = scheduler.timesteps
    if progress:
        iterator = tqdm(iterator, desc="DDPM sampling", leave=False)

    for t in iterator:
        if is_class_conditioned(model):
            noise_pred = model(sample, t, class_labels=class_labels).sample
        else:
            noise_pred = model(sample, t).sample
        sample = scheduler.step(noise_pred, t, sample, generator=generator).prev_sample

    return sample
