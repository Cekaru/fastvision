"""ToMe: iterative bipartite soft matching that *merges* instead of drops.

Tokens are split alternately into source/destination sets; each source token
is matched to its most similar destination and the r most similar pairs are
merged by size-weighted averaging. Repeats until the budget is met, so it
stays effective at aggressive ratios where a single bipartite round (which
can merge at most half the tokens) is not enough. Size tracking makes the
merge mass-conserving: ``sum(feats * sizes)`` is invariant.

Reference:
    Bolya, Fu, Dai, Zhang, Feichtenhofer, Hoffman. "Token Merging: Your ViT
    But Faster." ICLR 2023. arXiv:2210.09461. Independent reimplementation,
    adapted from the ViT setting to projected MLLM visual tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Pruner


class ToMeMerger(Pruner):
    def select(self, feats: torch.Tensor, keep: int, meta: dict | None = None) -> torch.Tensor:
        return self.reduce(feats, keep, meta)[1]

    @torch.no_grad()
    def reduce(
        self, feats: torch.Tensor, keep: int, meta: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feats.dim() != 3:
            raise ValueError(f"feats must be [B, N, D], got shape {tuple(feats.shape)}")
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        B, N, D = feats.shape
        device = feats.device
        keep = min(keep, N)
        if keep >= N:
            idx = torch.arange(N, device=device).expand(B, N).contiguous()
            return feats, idx

        x = feats.float()
        sizes = torch.ones(B, N, device=device)
        pos = torch.arange(N, device=device).expand(B, N).contiguous()
        cur = N
        expand = lambda i: i.unsqueeze(-1).expand(-1, -1, D)  # noqa: E731

        while cur > keep:
            a_x, b_x = x[:, 0::2], x[:, 1::2]
            a_s, b_s = sizes[:, 0::2], sizes[:, 1::2]
            a_p, b_p = pos[:, 0::2], pos[:, 1::2]
            r = min(cur - keep, a_x.shape[1])

            sim = F.normalize(a_x, dim=-1) @ F.normalize(b_x, dim=-1).transpose(1, 2)
            node_max, node_dst = sim.max(dim=-1)  # [B, Ma]
            order = node_max.argsort(dim=1, descending=True)
            merge_src, keep_src = order[:, :r], order[:, r:]
            dst = node_dst.gather(1, merge_src)  # [B, r]

            # size-weighted merge of chosen sources into their destinations
            src_mass = (a_x * a_s.unsqueeze(-1)).gather(1, expand(merge_src))
            w_b = (b_x * b_s.unsqueeze(-1)).scatter_add(1, expand(dst), src_mass)
            b_s = b_s.scatter_add(1, dst, a_s.gather(1, merge_src))
            b_x = w_b / b_s.unsqueeze(-1)

            x = torch.cat([a_x.gather(1, expand(keep_src)), b_x], dim=1)
            sizes = torch.cat([a_s.gather(1, keep_src), b_s], dim=1)
            pos = torch.cat([a_p.gather(1, keep_src), b_p], dim=1)
            cur -= r

        order = pos.argsort(dim=1)
        return x.gather(1, expand(order)).to(feats.dtype), pos.gather(1, order)
