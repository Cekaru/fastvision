"""FastV-style attention-score pruning at the feature boundary.

The original FastV ranks visual tokens by the attention they receive inside
early LLM layers, which requires ``attn_implementation="eager"`` (fused
kernels never materialize attention weights). This wrapper intercepts
*before* the LLM, so we score with the same quantity one step earlier: the
scaled dot product between a text query vector (mean prompt embedding,
supplied by the adapter in ``meta["query"]``) and each projected visual
token — both already live in the LLM input space thanks to the projector.
Attention-implementation-agnostic; falls back to the mean visual feature as
query when no text context is available.

Reference:
    Chen, Zhao, Liu, Bai, Lin, Zhou, Chang. "An Image is Worth 1/2 Tokens
    After Layer 2: Plug-and-Play Inference Acceleration for Large
    Vision-Language Models." ECCV 2024 (Oral). arXiv:2403.06764. This adapts
    FastV's attention-scoring idea to the pre-LLM feature boundary.
"""

from __future__ import annotations

import logging

import torch

from .base import Pruner

logger = logging.getLogger("fastvision")


class FastVPruner(Pruner):
    @torch.no_grad()
    def select(self, feats: torch.Tensor, keep: int, meta: dict | None = None) -> torch.Tensor:
        if feats.dim() != 3:
            raise ValueError(f"feats must be [B, N, D], got shape {tuple(feats.shape)}")
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        B, N, D = feats.shape
        keep = min(keep, N)
        query = (meta or {}).get("query")
        if query is None:
            logger.debug("fastvision: no meta['query']; scoring against mean visual feature")
            query = feats.float().mean(dim=1)  # [B, D]
        query = query.float().to(feats.device)
        if query.dim() == 1:
            query = query.expand(B, -1)
        if query.shape != (B, D):
            raise ValueError(
                f"meta['query'] must be [D] or [B, D] with B={B}, D={D}, "
                f"got shape {tuple(query.shape)}"
            )
        scores = torch.einsum("bnd,bd->bn", feats.float(), query) / D**0.5
        return scores.topk(keep, dim=1).indices.sort(dim=1).values
