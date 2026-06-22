"""Hugging Face Hub publishing: upload a trained model with a generated card.

Publishing is opt-in (``hub.push_to_hub: true``) and authenticated via the
``HF_TOKEN`` environment variable. If no token is present the functions log a
warning and return ``None`` instead of failing, so training never breaks just
because credentials are absent.
"""

from __future__ import annotations

import os
from pathlib import Path

from difflab.config import ExperimentConfig
from difflab.utils import get_logger

logger = get_logger()


def generate_model_card(cfg: ExperimentConfig, repo_id: str) -> str:
    """Render a Markdown model card describing how the model was produced."""
    task_titles = {
        "finetune": "Fine-tuned diffusion model",
        "class_conditioned": "Class-conditioned diffusion model",
        "audio": "Audio (Mel-spectrogram) diffusion model",
        "ddim_inversion": "DDIM inversion model",
    }
    title = task_titles.get(cfg.task, "Diffusion model")
    sched = cfg.scheduler
    tr = cfg.train
    return f"""---
license: mit
tags:
  - diffusion
  - {cfg.task}
  - difflab
library_name: diffusers
---

# {title} — `{repo_id}`

This model was trained with [`difflab`](https://github.com/ROHITCRAFTSYT/difflab), a
toolkit for diffusion models. It is a `UNet2DModel` trained with the DDPM
objective and can be sampled with either the DDPM or DDIM scheduler.

## Training summary

| Setting | Value |
| --- | --- |
| Task | `{cfg.task}` |
| Dataset | `{cfg.data.dataset}` |
| Image size | {cfg.model.sample_size}×{cfg.model.sample_size} |
| Channels | {cfg.model.in_channels} |
| Classes | {cfg.model.num_classes if cfg.model.num_classes else "unconditional"} |
| Train timesteps | {sched.num_train_timesteps} |
| Beta schedule | `{sched.beta_schedule}` |
| Prediction type | `{sched.prediction_type}` |
| Epochs | {tr.epochs} |
| Batch size | {tr.batch_size} |
| Learning rate | {tr.learning_rate} |
| EMA | {tr.use_ema} |

## Usage

```python
from diffusers import UNet2DModel, DDIMScheduler
from difflab.sampling import ddim_sample

model = UNet2DModel.from_pretrained("{repo_id}")
scheduler = DDIMScheduler(num_train_timesteps={sched.num_train_timesteps})
images = ddim_sample(model, scheduler, num_samples=8, num_inference_steps=50)
```

## Limitations

Sample quality depends on training budget. This card was generated automatically
from the training configuration.
"""


def push_model_to_hub(checkpoint_dir: str | Path, cfg: ExperimentConfig) -> str | None:
    """Upload ``checkpoint_dir`` (a saved UNet) to the Hub with a model card.

    Returns the repo URL on success, or ``None`` if pushing is disabled or no
    ``HF_TOKEN`` is configured.
    """
    if not cfg.hub.push_to_hub:
        logger.info("hub.push_to_hub is false; skipping upload.")
        return None

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set; skipping Hub upload.")
        return None

    repo_id = cfg.hub.repo_id
    if not repo_id:
        raise ValueError("hub.repo_id must be set when hub.push_to_hub is true.")

    from huggingface_hub import HfApi

    checkpoint_dir = Path(checkpoint_dir)
    card = generate_model_card(cfg, repo_id)
    (checkpoint_dir / "README.md").write_text(card, encoding="utf-8")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=cfg.hub.private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(checkpoint_dir), commit_message="Upload from difflab")
    url = f"https://huggingface.co/{repo_id}"
    logger.info("pushed model to %s", url)
    return url
