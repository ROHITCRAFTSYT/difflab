"""DDPM / DDIM sampling: shapes, determinism, conditioning, custom latents."""

from __future__ import annotations

import torch

from difflab.sampling import ddim_sample, ddpm_sample
from difflab.sampling.guidance import classifier_free_guidance, cond_uncond_labels


def test_ddpm_sample_shape(tiny_unet, ddpm_scheduler):
    out = ddpm_sample(tiny_unet, ddpm_scheduler, num_samples=2, num_inference_steps=5, device="cpu")
    assert out.shape == (2, 1, 8, 8)
    assert out.min() >= -1.0001 and out.max() <= 1.0001


def test_ddim_is_deterministic(tiny_unet, ddim_scheduler):
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = ddim_sample(tiny_unet, ddim_scheduler, 2, num_inference_steps=5, generator=g1, device="cpu")
    b = ddim_sample(tiny_unet, ddim_scheduler, 2, num_inference_steps=5, generator=g2, device="cpu")
    assert torch.allclose(a, b, atol=1e-5)


def test_ddim_accepts_custom_latents(tiny_unet, ddim_scheduler):
    latents = torch.randn(2, 1, 8, 8)
    out = ddim_sample(
        tiny_unet, ddim_scheduler, 2, num_inference_steps=4, latents=latents, device="cpu"
    )
    assert out.shape == latents.shape


def test_class_conditioned_sampling(tiny_cond_unet, ddpm_scheduler):
    labels = torch.tensor([0, 1, 2])
    out = ddpm_sample(
        tiny_cond_unet, ddpm_scheduler, 3, num_inference_steps=4, class_labels=labels, device="cpu"
    )
    assert out.shape == (3, 1, 8, 8)


def test_classifier_free_guidance_identity():
    uncond = torch.randn(2, 1, 4, 4)
    cond = torch.randn(2, 1, 4, 4)
    # scale=1 recovers the conditional prediction exactly.
    assert torch.allclose(classifier_free_guidance(uncond, cond, 1.0), cond)


def test_cond_uncond_labels_layout():
    labels = torch.tensor([2, 5])
    stacked = cond_uncond_labels(labels, null_label=0)
    assert stacked.tolist() == [0, 0, 2, 5]
