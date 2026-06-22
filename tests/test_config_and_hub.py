"""Config loading/validation and model-card generation (hermetic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from difflab.config import config_from_dict, load_config
from difflab.hub import generate_model_card, push_model_to_hub

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", [p.name for p in CONFIG_DIR.glob("*.yaml")])
def test_all_shipped_configs_load(name):
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.task in ("finetune", "class_conditioned", "audio", "ddim_inversion")


def test_class_conditioned_requires_num_classes():
    with pytest.raises(ValueError):
        config_from_dict({"task": "class_conditioned", "model": {"num_classes": 0}})


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"train": {"not_a_real_key": 1}})


def test_model_card_contains_key_facts():
    cfg = load_config(CONFIG_DIR / "class_conditioned_fashionmnist.yaml")
    card = generate_model_card(cfg, "user/my-model")
    assert "user/my-model" in card
    assert "fashion_mnist" in card
    assert "license: mit" in card


def test_push_disabled_returns_none(tmp_path):
    cfg = load_config(CONFIG_DIR / "class_conditioned_fashionmnist.yaml")
    assert cfg.hub.push_to_hub is False
    assert push_model_to_hub(tmp_path, cfg) is None
