"""difflab: a production-grade diffusion model toolkit.

Public API surface is intentionally small; import submodules directly for the
full feature set (training, sampling, inversion, audio).
"""

from __future__ import annotations

__version__ = "0.1.0"

from difflab.config import ExperimentConfig, load_config
from difflab.utils import get_device, make_image_grid, set_seed

__all__ = [
    "__version__",
    "ExperimentConfig",
    "load_config",
    "get_device",
    "set_seed",
    "make_image_grid",
]
