"""DivPrune: greedy Max-Min diversity (farthest-point) token selection.

Training-free and attention-free: operates purely on visual token features,
so it composes with FlashAttention / SDPA. O(N*K) per batch item, fully
vectorized across the batch.

Reference:
    Alvar, Singh, Akbari, Zhang. "DivPrune: Diversity-based Visual Token
    Pruning for Large Multimodal Models." CVPR 2025. arXiv:2503.02175.
    This is an independent reimplementation of the method described there.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Pruner


class DivPrune(Pruner):
    def __init__(self, distance: str = "cosine"):
        if distance not in ("cosine", "euclidean"):
            raise ValueError(f"distance must be 'cosine' or 'euclidean', got {distance!r}")
        self.distance = distance

    @torch.no_grad()
    def select(self, feats: torch.Tensor, keep: int, meta: dict | None = None) -> torch.Tensor:
        if feats.dim() != 3:
            raise ValueError(f"feats must be [B, N, D], got shape {tuple(feats.shape)}")
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        B, N, _ = feats.shape
        keep = min(keep, N)
        device = feats.device
        batch = torch.arange(B, device=device)

        x = feats.float()
        if self.distance == "cosine":
            base = F.normalize(x, dim=-1)

            def dist_to(idx: torch.Tensor) -> torch.Tensor:  # [B] -> [B, N]
                sel = base[batch, idx]
                return 1.0 - torch.einsum("bnd,bd->bn", base, sel)
        else:
            base = x

            def dist_to(idx: torch.Tensor) -> torch.Tensor:
                sel = base[batch, idx]
                return (base - sel[:, None, :]).pow(2).sum(-1)

        # Seed: highest-norm token per batch item.
        seed = x.norm(dim=-1).argmax(dim=1)
        selected = torch.empty(B, keep, dtype=torch.long, device=device)
        selected[:, 0] = seed
        min_dist = dist_to(seed)
        min_dist[batch, seed] = float("-inf")

        for i in range(1, keep):
            idx = min_dist.argmax(dim=1)
            selected[:, i] = idx
            min_dist = torch.minimum(min_dist, dist_to(idx))
            min_dist[batch, idx] = float("-inf")

        return selected.sort(dim=1).values
