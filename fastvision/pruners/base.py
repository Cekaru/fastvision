"""Pruner interface and budget resolution."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Pruner(ABC):
    """Selects which visual tokens to keep.

    Implementations must be deterministic given identical inputs (and seed,
    where applicable) and must return indices sorted ascending so that the
    spatial order of kept tokens is preserved in the sequence.
    """

    @abstractmethod
    def select(
        self,
        feats: torch.Tensor,
        keep: int,
        meta: dict | None = None,
    ) -> torch.Tensor:
        """Choose ``keep`` tokens per batch item.

        Args:
            feats: projected visual tokens, shape ``[B, N, D]``.
            keep: number of tokens to keep (``K <= N``).
            meta: optional adapter-provided context.

        Returns:
            keep_index: ``[B, K]`` long tensor, sorted ascending per row.
        """

    def reduce(
        self,
        feats: torch.Tensor,
        keep: int,
        meta: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(kept_features [B, K, D], keep_index [B, K])``.

        Default: gather at :meth:`select` indices. Mergers override this to
        return features that combine dropped tokens into the kept positions.
        """
        idx = self.select(feats, keep, meta)
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])), idx


def resolve_keep(
    n: int,
    keep_ratio: float | None = None,
    keep_tokens: int | None = None,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> int:
    """Resolve a token budget for ``n`` input tokens.

    ``keep_tokens`` (absolute) takes precedence over ``keep_ratio``
    (fraction). The result is clamped to ``[min_keep, max_keep]`` and never
    exceeds ``n``.
    """
    if keep_tokens is not None:
        k = int(keep_tokens)
    elif keep_ratio is not None:
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        k = max(1, round(n * keep_ratio))
    else:
        raise ValueError("one of keep_ratio or keep_tokens is required")
    k = max(k, min_keep)
    if max_keep is not None:
        k = min(k, max_keep)
    return min(k, n)
