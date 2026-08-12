"""Baseline pruners for ablation: random, uniform-stride, top-norm."""

from __future__ import annotations

import torch

from .base import Pruner


class RandomPruner(Pruner):
    def __init__(self, seed: int = 0):
        self.seed = seed

    @torch.no_grad()
    def select(self, feats: torch.Tensor, keep: int, meta: dict | None = None) -> torch.Tensor:
        B, N, _ = feats.shape
        keep = min(keep, N)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        rows = [torch.randperm(N, generator=gen)[:keep].sort().values for _ in range(B)]
        return torch.stack(rows).to(feats.device)


class UniformPruner(Pruner):
    @torch.no_grad()
    def select(self, feats: torch.Tensor, keep: int, meta: dict | None = None) -> torch.Tensor:
        B, N, _ = feats.shape
        keep = min(keep, N)
        idx = torch.linspace(0, N - 1, keep, device=feats.device).round().long()
        return idx.unsqueeze(0).expand(B, -1).contiguous()


class TopNormPruner(Pruner):
    @torch.no_grad()
    def select(self, feats: torch.Tensor, keep: int, meta: dict | None = None) -> torch.Tensor:
        _, N, _ = feats.shape
        keep = min(keep, N)
        norms = feats.float().norm(dim=-1)
        return norms.topk(keep, dim=1).indices.sort(dim=1).values
