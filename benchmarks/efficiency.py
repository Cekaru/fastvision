"""Prefill latency / throughput / memory measurement.

CUDA-event timing with warmup and median-of-N; falls back to wall clock on
CPU so the harness stays runnable everywhere.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import torch


@dataclass
class EfficiencyResult:
    prefill_latency_s: float  # median over iters
    prefill_tokens_per_s: float
    peak_memory_mb: float | None  # CUDA only
    seq_len: int

    def to_dict(self) -> dict:
        return asdict(self)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def measure_prefill(
    model, inputs: dict, warmup: int = 2, iters: int = 5
) -> EfficiencyResult:
    """Time a full multimodal prefill (single forward) on ``inputs``."""
    cuda = torch.cuda.is_available()
    for _ in range(warmup):
        model(**inputs)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    times = []
    seq_len = 0
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        out = model(**inputs)
        _sync()
        times.append(time.perf_counter() - t0)
        seq_len = out.logits.shape[1]

    latency = statistics.median(times)
    total_tokens = inputs["input_ids"].shape[0] * inputs["input_ids"].shape[1]
    return EfficiencyResult(
        prefill_latency_s=latency,
        prefill_tokens_per_s=total_tokens / latency if latency > 0 else 0.0,
        peak_memory_mb=torch.cuda.max_memory_allocated() / 1e6 if cuda else None,
        seq_len=seq_len,
    )
