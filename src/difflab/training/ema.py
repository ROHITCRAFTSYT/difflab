"""Exponential Moving Average of model weights.

Maintaining an EMA copy of the UNet weights and sampling from it is a standard
trick that noticeably improves diffusion sample quality. Kept dependency-free
and device-agnostic so it works identically on CPU and GPU.
"""

from __future__ import annotations

import copy

import torch


class EMA:
    """Tracks a shadow copy of a module's parameters via EMA updates."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Pull the shadow weights a fraction ``1 - decay`` toward ``model``."""
        for ema_p, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        # Buffers (e.g. norm stats) are copied verbatim.
        for ema_b, b in zip(self.shadow.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)

    @property
    def module(self) -> torch.nn.Module:
        return self.shadow
