"""DDIM inversion and prompt-based image editing.

DDIM sampling (``eta = 0``) is a deterministic ODE solver: given a noise latent
``x_T`` it integrates *down* to a clean latent ``x_0``. Because the map is
deterministic and (approximately) invertible, we can integrate the *same* ODE
*up* — from a real image's latent ``x_0`` back to the noise ``x_T`` that would
regenerate it. That noise is an editable handle on the image: re-running the
forward process with a modified prompt produces a targeted edit.

The core routines (:func:`ddim_invert`, :func:`ddim_sample_latents`) are written
against an arbitrary ``eps_fn(latent, t) -> noise_prediction`` so they can be
unit-tested on a tiny CPU model. :class:`DDIMInverter` wires them to a real
Stable Diffusion pipeline with classifier-free guidance.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from diffusers import DDIMScheduler
from tqdm.auto import tqdm

EpsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _alpha_at(scheduler: DDIMScheduler, timestep: int) -> torch.Tensor:
    """Cumulative alpha at ``timestep``; falls back to ``final_alpha_cumprod`` for t < 0."""
    if timestep < 0:
        return scheduler.final_alpha_cumprod
    return scheduler.alphas_cumprod[timestep]


@torch.no_grad()
def ddim_invert(
    eps_fn: EpsFn,
    scheduler: DDIMScheduler,
    latents: torch.Tensor,
    num_inference_steps: int = 50,
    progress: bool = False,
) -> torch.Tensor:
    """Invert clean ``latents`` (x_0) to the noise latent (x_T).

    Implements the deterministic DDIM forward (inversion) recurrence: at each
    step it estimates x_0, then advances to the *next, noisier* timestep.
    """
    device = latents.device
    scheduler.set_timesteps(num_inference_steps, device=device)
    step = scheduler.config.num_train_timesteps // num_inference_steps

    latent = latents.clone()
    # Walk timesteps from low noise to high noise (reverse of sampling order).
    timesteps = list(reversed(scheduler.timesteps.tolist()))
    iterator = tqdm(timesteps, desc="DDIM inversion", leave=False) if progress else timesteps

    for t in iterator:
        t_next = min(t + step, scheduler.config.num_train_timesteps - 1)
        eps = eps_fn(latent, torch.tensor(t, device=device))
        alpha_t = _alpha_at(scheduler, t)
        alpha_next = _alpha_at(scheduler, t_next)
        # Predict x_0 from the current (t) latent, then re-noise to t_next.
        x0 = (latent - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
        latent = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * eps
    return latent


@torch.no_grad()
def ddim_sample_latents(
    eps_fn: EpsFn,
    scheduler: DDIMScheduler,
    latents: torch.Tensor,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    progress: bool = False,
) -> torch.Tensor:
    """Deterministically integrate noise ``latents`` (x_T) down to x_0."""
    device = latents.device
    scheduler.set_timesteps(num_inference_steps, device=device)
    latent = latents.clone()
    iterator = scheduler.timesteps
    if progress:
        iterator = tqdm(iterator, desc="DDIM sampling", leave=False)
    for t in iterator:
        eps = eps_fn(latent, t)
        latent = scheduler.step(eps, t, latent, eta=eta).prev_sample
    return latent


class DDIMInverter:
    """Real-image inversion + prompt editing on a Stable Diffusion pipeline.

    Example
    -------
    >>> inv = DDIMInverter.from_pretrained("runwayml/stable-diffusion-v1-5")
    >>> latents = inv.invert(image, prompt="a photo of a cat")
    >>> edited = inv.sample(latents, prompt="a photo of a dog")  # edit
    """

    def __init__(self, pipe, num_inference_steps: int = 50, guidance_scale: float = 1.0):
        self.pipe = pipe
        self.device = pipe.device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        # Inversion requires a DDIM scheduler with x0 clipping disabled, else the
        # deterministic forward/backward steps stop being exact inverses.
        self.scheduler = DDIMScheduler.from_config(
            {**dict(pipe.scheduler.config), "clip_sample": False, "set_alpha_to_one": False}
        )

    @classmethod
    def from_pretrained(cls, model_id: str, device: str | None = None, **kw):
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
        if device:
            pipe = pipe.to(device)
        return cls(pipe, **kw)

    @torch.no_grad()
    def _text_embeddings(self, prompt: str) -> torch.Tensor:
        tok = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.pipe.text_encoder(tok.input_ids.to(self.device))[0]

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        """Encode a PIL image (or [-1,1] tensor) to scaled VAE latents."""
        if not torch.is_tensor(image):
            import numpy as np

            arr = np.asarray(image.convert("RGB"), dtype="float32") / 127.5 - 1.0
            image = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device, dtype=self.pipe.vae.dtype)
        posterior = self.pipe.vae.encode(image).latent_dist
        return posterior.mean * self.pipe.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return self.pipe.numpy_to_pil(image.permute(0, 2, 3, 1).cpu().numpy())

    def _make_eps_fn(self, prompt: str) -> EpsFn:
        """Build an ``eps_fn`` for ``prompt`` with classifier-free guidance."""
        cond = self._text_embeddings(prompt)
        if self.guidance_scale > 1.0:
            uncond = self._text_embeddings("")
            embeddings = torch.cat([uncond, cond])
        else:
            embeddings = cond

        def eps_fn(latent: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if self.guidance_scale > 1.0:
                model_in = torch.cat([latent] * 2)
                noise = self.pipe.unet(model_in, t, encoder_hidden_states=embeddings).sample
                noise_uncond, noise_cond = noise.chunk(2)
                return noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            return self.pipe.unet(latent, t, encoder_hidden_states=embeddings).sample

        return eps_fn

    @torch.no_grad()
    def invert(self, image, prompt: str = "") -> torch.Tensor:
        """Invert a real image to its noise latent under ``prompt``."""
        latents = self.encode_image(image)
        return ddim_invert(self._make_eps_fn(prompt), self.scheduler, latents, self.num_inference_steps)

    @torch.no_grad()
    def sample(self, latents: torch.Tensor, prompt: str):
        """Regenerate (or edit) from noise ``latents`` under ``prompt``."""
        out = ddim_sample_latents(
            self._make_eps_fn(prompt), self.scheduler, latents, self.num_inference_steps
        )
        return self.decode_latents(out)
