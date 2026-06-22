#!/usr/bin/env python
"""Thin wrapper around ``difflab sample``.

    python scripts/sample.py -c configs/class_conditioned_fashionmnist.yaml \
        --checkpoint outputs/class_cond_fashionmnist/final --num 16
"""

from __future__ import annotations

import sys

from difflab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["sample", *sys.argv[1:]]))
