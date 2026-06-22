"""DDIM inversion math: the invert -> sample round-trip is the identity.

DDIM inversion and DDIM sampling are exact inverses *when the noise prediction
is consistent across adjacent timesteps* (the standard assumption that holds for
a smooth, trained model). We therefore verify the recurrence with a constant
``eps_fn`` (independent of latent and t), where the inverse property is exact up
to floating point — this isolates and validates our implementation of the
forward/backward DDIM steps without needing a trained Stable Diffusion model.

A second test confirms the realistic behaviour: with a generic (untrained)
model the round-trip is only approximate, exactly as theory predicts.
"""

from __future__ import annotations

import torch

from difflab.inversion.ddim_inversion import ddim_invert, ddim_sample_latents


def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item() / b.norm().item()


def test_roundtrip_exact_for_constant_eps(ddim_inv_scheduler):
    """With a consistent eps prediction, invert then sample is the identity."""
    torch.manual_seed(0)
    x0 = torch.randn(2, 1, 8, 8)
    const = torch.randn(1, 1, 8, 8)  # same prediction at every step

    def eps_fn(latent, t):
        return const.expand_as(latent)

    noise = ddim_invert(eps_fn, ddim_inv_scheduler, x0, num_inference_steps=50)
    recon = ddim_sample_latents(eps_fn, ddim_inv_scheduler, noise, num_inference_steps=50)
    assert _rel_error(recon, x0) < 1e-4


def test_zero_eps_roundtrip(ddim_inv_scheduler):
    x0 = torch.randn(1, 1, 8, 8)
    eps_fn = lambda latent, t: torch.zeros_like(latent)  # noqa: E731
    noise = ddim_invert(eps_fn, ddim_inv_scheduler, x0, num_inference_steps=25)
    recon = ddim_sample_latents(eps_fn, ddim_inv_scheduler, noise, num_inference_steps=25)
    # ~1% residual from endpoint discretization (x0 vs the smallest sampling timestep).
    assert _rel_error(recon, x0) < 0.05


def test_generic_model_roundtrip_is_approximate(tiny_unet, ddim_inv_scheduler):
    """A generic untrained model gives a non-identity (but finite) round-trip."""
    x0 = torch.randn(1, 1, 8, 8)
    eps_fn = lambda latent, t: tiny_unet(latent, t).sample  # noqa: E731
    noise = ddim_invert(eps_fn, ddim_inv_scheduler, x0, num_inference_steps=50)
    recon = ddim_sample_latents(eps_fn, ddim_inv_scheduler, noise, num_inference_steps=50)
    err = _rel_error(recon, x0)
    assert 0.0 < err < 2.0  # finite and not NaN; quality depends on the model


def test_inversion_shape_preserved(tiny_unet, ddim_inv_scheduler):
    x0 = torch.randn(3, 1, 8, 8)
    noise = ddim_invert(lambda x, t: tiny_unet(x, t).sample, ddim_inv_scheduler, x0, 20)
    assert noise.shape == x0.shape
