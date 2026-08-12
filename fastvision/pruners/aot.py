"""Experimental optimal-transport token merger (pure torch, no POT dep).

Anchors are chosen with farthest-point sampling (DivPrune), then an
entropy-regularized transport plan between all N tokens and the K anchors is
computed with Sinkhorn iterations on a cosine cost. Each output token is the
transport-weighted barycenter of its incoming mass, so dropped tokens
contribute to the nearest anchors instead of vanishing.

References:
    Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal
    Transport." NeurIPS 2013. arXiv:1306.0895 (Sinkhorn iteration).
    Anchor selection reuses DivPrune (Alvar et al., CVPR 2025,
    arXiv:2503.02175). This optimal-transport merger is original to
    FastVision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Pruner
from .divprune import DivPrune


class OTMerger(Pruner):
    def __init__(self, epsilon: float = 0.05, iters: int = 20):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if iters < 1:
            raise ValueError(f"iters must be >= 1, got {iters}")
        self.epsilon = epsilon
        self.iters = iters
        self._anchors = DivPrune()

    def select(self, feats: torch.Tensor, keep: int, meta: dict | None = None) -> torch.Tensor:
        return self._anchors.select(feats, keep, meta)

    @torch.no_grad()
    def reduce(
        self, feats: torch.Tensor, keep: int, meta: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feats.dim() != 3:
            raise ValueError(f"feats must be [B, N, D], got shape {tuple(feats.shape)}")
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        B, N, D = feats.shape
        keep = min(keep, N)
        idx = self.select(feats, keep, meta)
        if keep >= N:
            return feats, idx

        x = F.normalize(feats.float(), dim=-1)
        anchors = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        cost = 1.0 - x @ anchors.transpose(1, 2)  # [B, N, K]

        # Sinkhorn: uniform source mass over N, uniform target mass over K
        log_k = -cost / self.epsilon
        log_mu = torch.full((B, N, 1), -torch.log(torch.tensor(float(N))), device=x.device)
        log_nu = torch.full((B, 1, keep), -torch.log(torch.tensor(float(keep))), device=x.device)
        log_u = torch.zeros(B, N, 1, device=x.device)
        log_v = torch.zeros(B, 1, keep, device=x.device)
        for _ in range(self.iters):
            log_u = log_mu - torch.logsumexp(log_k + log_v, dim=2, keepdim=True)
            log_v = log_nu - torch.logsumexp(log_k + log_u, dim=1, keepdim=True)
        plan = torch.exp(log_k + log_u + log_v)  # [B, N, K], columns sum ~1/K

        merged = plan.transpose(1, 2) @ feats.float()  # [B, K, D]
        merged = merged / plan.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
        return merged.to(feats.dtype), idx
