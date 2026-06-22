# Configuration

Every run is described by a YAML file deserialized into a validated
`ExperimentConfig` (see `src/difflab/config.py`). Unknown keys are rejected, so
typos fail fast.

## Top-level

| Field | Meaning |
| --- | --- |
| `task` | One of `finetune`, `class_conditioned`, `audio`, `ddim_inversion`. |
| `name` | Run name (used for logging / Hub repo). |
| `model` | UNet / pipeline architecture. |
| `scheduler` | Noise schedule. |
| `data` | Dataset + preprocessing. |
| `train` | Optimization loop. |
| `hub` | Hugging Face Hub publishing. |

## `model`

| Field | Default | Notes |
| --- | --- | --- |
| `sample_size` | 32 | Image side length. |
| `in_channels` / `out_channels` | 1 | 1 for grayscale/audio, 3 for RGB. |
| `layers_per_block` | 2 | UNet depth per resolution. |
| `norm_num_groups` | 32 | Must divide every `block_out_channels` entry. |
| `block_out_channels` | `[64,128,128,256]` | Channels per resolution. |
| `down_block_types` / `up_block_types` | — | Must match `block_out_channels` length. |
| `num_classes` | 0 | `>0` enables class conditioning. |
| `pretrained` | `null` | Hub id for fine-tuning / inversion. |

## `scheduler`

| Field | Default |
| --- | --- |
| `num_train_timesteps` | 1000 |
| `beta_schedule` | `linear` (`scaled_linear`, `squaredcos_cap_v2`) |
| `beta_start` / `beta_end` | `1e-4` / `2e-2` |
| `prediction_type` | `epsilon` (`v_prediction`, `sample`) |

## `data`

Image keys: `dataset`, `split`, `image_column`, `label_column`, `resolution`,
`center_crop`, `random_flip`, `max_samples`.
Audio keys: `audio_column`, `sample_rate`, `n_fft`, `hop_length`, `n_mels`,
`audio_slice_seconds`.

## `train`

`output_dir`, `epochs`, `max_steps`, `batch_size`,
`gradient_accumulation_steps`, `learning_rate`, `lr_warmup_steps`,
`mixed_precision` (`no`/`fp16`/`bf16`), `use_ema`, `ema_decay`,
`save_every_epochs`, `sample_every_epochs`, `num_eval_samples`,
`num_inference_steps`, `seed`, `dataloader_num_workers`.

`max_steps` caps optimizer steps regardless of `epochs` — this is what makes the
`*_smoke.yaml` configs finish in seconds.

## `hub`

| Field | Default | Notes |
| --- | --- | --- |
| `push_to_hub` | `false` | Opt-in. |
| `repo_id` | `null` | Required when pushing. |
| `private` | `false` | |

Authentication is via the `HF_TOKEN` environment variable; if it is missing the
upload is skipped with a warning rather than failing.
