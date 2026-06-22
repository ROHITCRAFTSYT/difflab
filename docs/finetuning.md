# Fine-tuning

Fine-tuning continues training a **pretrained** diffusion UNet on a new dataset.
It is far cheaper than training from scratch and is the recommended way to
specialise a model to a new visual domain.

## How it works

1. Load a pretrained `UNet2DModel` (`model.pretrained`, a Hub id).
2. The loaded model's spatial size and channel count override the config so
   sampling matches the checkpoint.
3. Train with the shared [`Trainer`](configuration.md) on the target dataset.

Entry point: `difflab.training.finetune.run`.

## Run it

Smoke test (CPU, downloads a small pretrained UNet + a few images):

```bash
difflab train -c configs/finetune_butterflies_smoke.yaml
```

Full run (GPU):

```bash
difflab train -c configs/finetune_butterflies.yaml
```

Then sample:

```bash
difflab sample -c configs/finetune_butterflies.yaml \
    --checkpoint outputs/finetune_butterflies/final --num 16
```

## Tips

- Use a **smaller learning rate** than from-scratch training (the default config
  uses `5e-5`) to avoid catastrophic forgetting.
- Enable **EMA** (`train.use_ema: true`) for smoother samples.
- `random_flip: true` is a cheap, safe augmentation for most natural-image data.

See the notebook [`01_finetuning.ipynb`](https://github.com/ROHITCRAFTSYT/difflab/blob/main/notebooks/01_finetuning.ipynb).
