"""Lightweight timing/memory probes for profile mode."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch


@contextmanager
def timed(stats: dict, key: str):
    """Record wall-clock seconds (CUDA-synchronized when available) into stats."""
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize()
        stats[key] = time.perf_counter() - t0


def peak_memory_mb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return None
