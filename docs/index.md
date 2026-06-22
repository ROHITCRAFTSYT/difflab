# difflab

A production-grade toolkit for **diffusion models**, packaged as a small,
well-tested Python library. It implements four capabilities behind one
config-driven interface:

- **Fine-tuning** — adapt a pretrained image diffusion model to a new dataset.
- **Class-conditioned generation** — train a label-conditioned UNet and sample
  any class on demand.
- **DDIM inversion** — deterministically invert a real image to its latent and
  edit it with a new prompt.
- **Audio diffusion** — generate audio by diffusing Mel spectrograms and
  reconstructing the waveform.

## Design principles

1. **One training loop.** Every image pillar shares
   [`Trainer`](https://github.com/ROHITCRAFTSYT/difflab) — the same DDPM objective,
   EMA, checkpointing, and sampling code. Pillars only differ in how they build
   the model and data.
2. **Config-driven.** A run is a single YAML file validated into a typed
   `ExperimentConfig`. Every experiment ships a `*_smoke.yaml` that runs in
   seconds on CPU.
3. **Verifiable without a GPU.** The test suite and smoke configs prove
   correctness on CPU; full-quality results come from a GPU notebook.
4. **Built on `diffusers`.** We use battle-tested `UNet2DModel` and schedulers,
   wrapped in our own readable training/sampling/inversion code.

## Where to start

- Read the [Theory](theory.md) page for the DDPM/DDIM math the code implements.
- Follow a pillar guide: [Fine-tuning](finetuning.md),
  [Class-conditioned](class_conditioning.md),
  [DDIM inversion](ddim_inversion.md), [Audio](audio.md).
- See [Configuration](configuration.md) for the full config schema.

## Install

```bash
pip install -e ".[dev]"   # CPU is fine for smoke tests and the test suite
pytest -q                 # prove the stack works
```
