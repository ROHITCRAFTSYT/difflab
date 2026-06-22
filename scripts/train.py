#!/usr/bin/env python
"""Thin wrapper around ``difflab train`` for users who prefer ``python scripts/train.py``.

    python scripts/train.py -c configs/class_conditioned_fashionmnist.yaml
"""

from __future__ import annotations

import sys

from difflab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
