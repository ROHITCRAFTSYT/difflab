"""Command-line interface: ``difflab {train,sample,invert}``.

All commands are config-driven. ``train`` dispatches to the right pillar based
on ``task`` in the YAML; ``sample`` loads a trained checkpoint and writes a grid;
``invert`` runs Stable Diffusion DDIM inversion + prompt editing on an image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from difflab.config import load_config
from difflab.utils import ensure_dir, get_logger

logger = get_logger()


def _dispatch_train(cfg):
    """Route a config to its pillar training entry point."""
    if cfg.task == "finetune":
        from difflab.training import finetune

        return finetune.run(cfg)
    if cfg.task == "class_conditioned":
        from difflab.training import class_conditioned

        return class_conditioned.run(cfg)
    if cfg.task == "audio":
        from difflab.training import audio

        return audio.run(cfg)
    raise ValueError(f"Task {cfg.task!r} is not trainable via the CLI.")


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result = _dispatch_train(cfg)
    logger.info(
        "training complete: %d steps, final loss %.4f, checkpoint -> %s",
        result.global_step,
        result.final_loss,
        result.checkpoint_dir,
    )
    if cfg.hub.push_to_hub:
        from difflab.hub import push_model_to_hub

        push_model_to_hub(result.checkpoint_dir, cfg)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    import torch
    from diffusers import UNet2DModel

    from difflab.models import build_scheduler
    from difflab.sampling import ddim_sample
    from difflab.utils import get_device, make_image_grid, tensor_to_pil

    cfg = load_config(args.config)
    device = get_device()
    model = UNet2DModel.from_pretrained(args.checkpoint).to(device)
    scheduler = build_scheduler(cfg.scheduler, kind="ddim")

    labels = None
    if cfg.model.num_classes > 0 and args.labels:
        labels = torch.tensor([int(x) for x in args.labels.split(",")], device=device)
        args.num = len(labels)

    images = ddim_sample(
        model,
        scheduler,
        num_samples=args.num,
        num_inference_steps=args.steps,
        class_labels=labels,
        device=device,
        progress=True,
    )
    grid = make_image_grid(tensor_to_pil(images))
    out = ensure_dir(args.out) / "sample_grid.png"
    grid.save(out)
    logger.info("wrote %s", out)
    return 0


def cmd_invert(args: argparse.Namespace) -> int:
    from PIL import Image

    from difflab.inversion import DDIMInverter
    from difflab.utils import get_device

    device = str(get_device())
    inv = DDIMInverter.from_pretrained(
        args.model, device=device, num_inference_steps=args.steps, guidance_scale=args.guidance
    )
    image = Image.open(args.image).convert("RGB").resize((512, 512))
    latents = inv.invert(image, prompt=args.source_prompt)
    edited = inv.sample(latents, prompt=args.target_prompt or args.source_prompt)
    out = ensure_dir(args.out) / "edited.png"
    edited[0].save(out)
    logger.info("wrote %s", out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="difflab", description="Diffusion model toolkit.")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train/fine-tune a model from a config.")
    t.add_argument("-c", "--config", required=True, type=Path)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("sample", help="Sample from a trained checkpoint.")
    s.add_argument("-c", "--config", required=True, type=Path)
    s.add_argument("--checkpoint", required=True, help="Path to a saved UNet directory.")
    s.add_argument("--num", type=int, default=16)
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--labels", default=None, help="Comma-separated class labels (conditioned models).")
    s.add_argument("--out", default="samples")
    s.set_defaults(func=cmd_sample)

    i = sub.add_parser("invert", help="DDIM-invert a real image and edit via a prompt.")
    i.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    i.add_argument("--image", required=True)
    i.add_argument("--source-prompt", default="")
    i.add_argument("--target-prompt", default=None)
    i.add_argument("--steps", type=int, default=50)
    i.add_argument("--guidance", type=float, default=1.0)
    i.add_argument("--out", default="edits")
    i.set_defaults(func=cmd_invert)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
