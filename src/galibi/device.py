"""Device selection.

The real runs are all CUDA. This exists so the training and evaluation loops can be
exercised end-to-end on a laptop with a tiny model before a pod is launched: loop bugs
are much cheaper to find at $0/hr than at $1.99/hr, and this project has already paid
for several of them at the higher rate.
"""

from __future__ import annotations

import os


def pick_device() -> str:
    override = os.environ.get("GALIBI_DEVICE")
    if override:
        return override
    import torch

    if torch.cuda.is_available():
        return "cuda"
    # MPS is deliberately not chosen automatically: several ops used here silently
    # fall back or differ numerically, which would make a smoke test that passes on a
    # laptop mean less than it appears to. Ask for it explicitly if you want it.
    return "cpu"


def empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
