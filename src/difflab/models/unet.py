"""UNet construction.

We use ``diffusers.UNet2DModel`` as the backbone. The same factory builds both
unconditional models (for fine-tuning / audio) and class-conditioned models by
toggling ``num_class_embeds``.
"""

from __future__ import annotations

from diffusers import UNet2DModel

from difflab.config import ModelConfig


def build_unet(cfg: ModelConfig) -> UNet2DModel:
    """Build a fresh ``UNet2DModel`` from a :class:`ModelConfig`.

    If ``cfg.num_classes > 0`` the model gets a learned class-embedding table
    (``num_class_embeds``), so ``model(sample, t, class_labels=...)`` conditions
    generation on integer labels. Otherwise the model is unconditional.
    """
    if len(cfg.down_block_types) != len(cfg.block_out_channels):
        raise ValueError(
            "down_block_types and block_out_channels must have equal length: "
            f"{len(cfg.down_block_types)} vs {len(cfg.block_out_channels)}"
        )
    if len(cfg.up_block_types) != len(cfg.block_out_channels):
        raise ValueError(
            "up_block_types and block_out_channels must have equal length: "
            f"{len(cfg.up_block_types)} vs {len(cfg.block_out_channels)}"
        )
    bad = [c for c in cfg.block_out_channels if c % cfg.norm_num_groups != 0]
    if bad:
        raise ValueError(
            f"Every block_out_channels entry must be divisible by norm_num_groups "
            f"({cfg.norm_num_groups}); offending channels: {bad}"
        )

    return UNet2DModel(
        sample_size=cfg.sample_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        layers_per_block=cfg.layers_per_block,
        block_out_channels=tuple(cfg.block_out_channels),
        down_block_types=tuple(cfg.down_block_types),
        up_block_types=tuple(cfg.up_block_types),
        norm_num_groups=cfg.norm_num_groups,
        num_class_embeds=cfg.num_classes if cfg.num_classes > 0 else None,
    )


def load_pretrained_unet(pretrained: str) -> UNet2DModel:
    """Load a pretrained ``UNet2DModel`` (the UNet sub-folder of a DDPM pipeline)."""
    return UNet2DModel.from_pretrained(pretrained)


def is_class_conditioned(model: UNet2DModel) -> bool:
    """True if the UNet expects ``class_labels`` at call time."""
    return getattr(model.config, "num_class_embeds", None) not in (None, 0)
