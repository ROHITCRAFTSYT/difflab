"""Guidance helpers for conditioned sampling.

Provides two complementary tools:

* :func:`classifier_free_guidance` — combine a conditional and unconditional
  noise prediction into a guided prediction. This is the standard mechanism
  behind text-to-image guidance scale.
* :func:`cond_uncond_labels` — build paired class-label batches for a
  class-conditioned ``UNet2DModel`` trained with label dropout, so the same
  model can be used unconditionally (null label) and conditionally.
"""

from __future__ import annotations

import torch


def classifier_free_guidance(
    noise_uncond: torch.Tensor,
    noise_cond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Apply classifier-free guidance.

    ``pred = uncond + scale * (cond - uncond)``. ``scale = 1`` recovers the plain
    conditional prediction; larger values push samples harder toward the
    condition at some cost to diversity.
    """
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


def cond_uncond_labels(
    class_labels: torch.Tensor,
    null_label: int,
) -> torch.Tensor:
    """Stack [unconditional || conditional] label batches for CFG.

    Expects a model trained with a reserved ``null_label`` (label dropout). The
    returned tensor has twice the batch size: the first half is all ``null_label``
    and the second half is the real labels, matching a duplicated latent batch.
    """
    uncond = torch.full_like(class_labels, null_label)
    return torch.cat([uncond, class_labels], dim=0)
