# DDIM inversion

DDIM inversion recovers the noise latent that deterministically regenerates a
**real** image, giving an editable handle on it. Re-sampling that latent under a
modified prompt produces a targeted edit while preserving structure.

See [Theory](theory.md) for the derivation.

## API

The core routines are model-agnostic and take any `eps_fn(latent, t)`:

```python
from difflab.inversion import ddim_invert, ddim_sample_latents
noise = ddim_invert(eps_fn, scheduler, latents, num_inference_steps=50)
recon = ddim_sample_latents(eps_fn, scheduler, noise, num_inference_steps=50)
```

`DDIMInverter` wires this to a Stable Diffusion pipeline with classifier-free
guidance and VAE encode/decode:

```python
from difflab.inversion import DDIMInverter
from PIL import Image

inv = DDIMInverter.from_pretrained("runwayml/stable-diffusion-v1-5", device="cuda")
image = Image.open("cat.png").convert("RGB").resize((512, 512))

latents = inv.invert(image, prompt="a photo of a cat")
edited  = inv.sample(latents, prompt="a photo of a dog")   # cat -> dog
edited[0].save("dog.png")
```

Or from the CLI:

```bash
difflab invert --image cat.png \
    --source-prompt "a photo of a cat" \
    --target-prompt "a photo of a dog" \
    --steps 50 --guidance 1.0
```

## Why `clip_sample=False`

Inversion is only exact when the forward and backward DDIM steps are true
inverses. The default `DDIMScheduler` clips the predicted $\hat x_0$ to
$[-1, 1]$, which breaks that symmetry. `DDIMInverter` therefore rebuilds the
scheduler with `clip_sample=False` (and `set_alpha_to_one=False`). The test
`test_inversion.py::test_zero_eps_roundtrip` fails without this fix — it is the
single most common DDIM-inversion bug.

## Practical notes

- More steps → lower reconstruction error (the approximation tightens).
- Keep `guidance_scale` low (≈1) during inversion; high guidance degrades
  reconstruction. Raise it only on the editing pass if needed.

See the notebook [`03_ddim_inversion.ipynb`](https://github.com/ROHITCRAFTSYT/difflab/blob/main/notebooks/03_ddim_inversion.ipynb).
