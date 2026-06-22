"""Deterministic DDIM sampling for ``UNet2DModel`` backbones.

DDIM (Song et al., 2021) defines a non-Markovian sampling process that, with
``eta = 0``, is fully deterministic given the initial noise. This determinism is
what makes DDIM *inversion* possible (see ``difflab.inversion``).
"""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler
from tqdm.auto import tqdm

from difflab.models.unet import is_class_conditioned


@torch.no_grad()
def ddim_sample(
    model,
    scheduler: DDIMScheduler,
    num_samples: int,
    image_size: int | None = None,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    class_labels: torch.Tensor | None = None,
    latents: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
) -> torch.Tensor:
    """Draw samples with the DDIM sampler.

    ``eta = 0`` gives deterministic sampling (the default). Pass ``latents`` to
    start from specific noise (e.g. the output of DDIM inversion). Returns a
    tensor in [-1, 1] of shape (num_samples, C, H, W).
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()

    channels = model.config.in_channels
    size = image_size or model.config.sample_size
    scheduler.set_timesteps(num_inference_steps, device=device)

    if is_class_conditioned(model):
        num_classes = model.config.num_class_embeds
        if class_labels is None:
            class_labels = torch.arange(num_samples, device=device) % num_classes
        class_labels = class_labels.to(device=device, dtype=torch.long)

    if latents is None:
        sample = torch.randn(
            (num_samples, channels, size, size), generator=generator, device=device
        )
    else:
        sample = latents.to(device)

    iterator = scheduler.timesteps
    if progress:
        iterator = tqdm(iterator, desc="DDIM sampling", leave=False)

    for t in iterator:
        if is_class_conditioned(model):
            noise_pred = model(sample, t, class_labels=class_labels).sample
        else:
            noise_pred = model(sample, t).sample
        sample = scheduler.step(noise_pred, t, sample, eta=eta, generator=generator).prev_sample

    return sample
