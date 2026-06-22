"""Typed, validated configuration schema for every experiment.

Each run is described by a single YAML file that deserializes into an
:class:`ExperimentConfig`. The schema is plain dataclasses (no heavy
dependencies) with explicit validation so misconfigured runs fail fast and
loudly rather than deep inside a training loop.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Tasks understood by the CLI / trainer dispatch.
TASKS = ("finetune", "class_conditioned", "audio", "ddim_inversion")


@dataclass
class ModelConfig:
    """UNet / pipeline architecture settings."""

    # For from-scratch models (class-conditioned, audio).
    sample_size: int = 32
    in_channels: int = 1
    out_channels: int = 1
    layers_per_block: int = 2
    norm_num_groups: int = 32  # GroupNorm groups; must divide every block_out_channels entry
    block_out_channels: tuple[int, ...] = (64, 128, 128, 256)
    down_block_types: tuple[str, ...] = (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    # Class-conditioning: number of label classes (0 disables conditioning).
    num_classes: int = 0
    # For fine-tuning / inversion: a pretrained model id on the Hub.
    pretrained: str | None = None


@dataclass
class SchedulerConfig:
    """Noise scheduler settings shared by training and sampling."""

    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"  # linear | scaled_linear | squaredcos_cap_v2
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    prediction_type: str = "epsilon"  # epsilon | v_prediction | sample


@dataclass
class DataConfig:
    """Dataset + preprocessing settings."""

    dataset: str = "fashion_mnist"  # HF datasets id or local path
    split: str = "train"
    image_column: str = "image"
    label_column: str | None = None
    resolution: int = 32
    center_crop: bool = True
    random_flip: bool = True
    # Cap dataset size (useful for smoke runs); None means use everything.
    max_samples: int | None = None
    # Stream `max_samples` examples instead of downloading the whole dataset.
    # Useful for large (audio) datasets; requires max_samples to be set.
    streaming: bool = False

    # Audio-only knobs (ignored for image pillars).
    audio_column: str = "audio"
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    audio_slice_seconds: float = 5.0


@dataclass
class TrainConfig:
    """Optimization / loop settings."""

    output_dir: str = "outputs/run"
    epochs: int = 50
    max_steps: int | None = None  # hard cap on optimizer steps (smoke runs)
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = "no"  # no | fp16 | bf16
    use_ema: bool = False
    ema_decay: float = 0.9999
    save_every_epochs: int = 10
    sample_every_epochs: int = 5
    num_eval_samples: int = 16
    num_inference_steps: int = 50
    seed: int = 0
    dataloader_num_workers: int = 0


@dataclass
class HubConfig:
    """Hugging Face Hub publishing settings (no-op without a token)."""

    push_to_hub: bool = False
    repo_id: str | None = None
    private: bool = False
    # Token is read from HF_TOKEN env var at push time; never stored in config.


@dataclass
class ExperimentConfig:
    """Top-level config for a single experiment."""

    task: str = "class_conditioned"
    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r}; expected one of {TASKS}.")
        if self.train.batch_size < 1:
            raise ValueError("train.batch_size must be >= 1.")
        if self.train.gradient_accumulation_steps < 1:
            raise ValueError("train.gradient_accumulation_steps must be >= 1.")
        if self.scheduler.prediction_type not in ("epsilon", "v_prediction", "sample"):
            raise ValueError(f"Invalid prediction_type {self.scheduler.prediction_type!r}.")
        if self.task == "class_conditioned" and self.model.num_classes <= 0:
            raise ValueError("class_conditioned task requires model.num_classes > 0.")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build_section(cls: type, payload: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass section, tolerating extra/missing keys gracefully."""
    payload = dict(payload or {})
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - field_names
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    # Coerce list -> tuple for tuple-typed fields so YAML lists round-trip cleanly.
    for f in dataclasses.fields(cls):
        if f.name in payload and isinstance(payload[f.name], list):
            payload[f.name] = tuple(payload[f.name])
    return cls(**payload)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a plain dict (e.g. parsed YAML)."""
    raw = dict(raw or {})
    return ExperimentConfig(
        task=raw.get("task", "class_conditioned"),
        name=raw.get("name", "experiment"),
        model=_build_section(ModelConfig, raw.get("model")),
        scheduler=_build_section(SchedulerConfig, raw.get("scheduler")),
        data=_build_section(DataConfig, raw.get("data")),
        train=_build_section(TrainConfig, raw.get("train")),
        hub=_build_section(HubConfig, raw.get("hub")),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}.")
    return config_from_dict(raw)
