"""FastVisionWrapper: install visual-token pruning on a HF multimodal model.

``FastVisionWrapper(model, ...)`` returns the *same* model object with hooks
installed, so existing ``.generate()`` code is untouched. ``model.unwrap()``
restores original behavior; ``model.fastvision.stats`` exposes last-run
token counts.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch.nn as nn

from .adapters.registry import detect_adapter
from .pruners import PRUNERS, Pruner, resolve_keep

logger = logging.getLogger("fastvision")


class FastVisionState:
    """Per-model pruning configuration and runtime stats."""

    def __init__(
        self,
        model: nn.Module,
        keep_ratio: float | None = 0.1,
        keep_tokens: int | None = None,
        strategy: str | Pruner = "divprune",
        distance: str = "cosine",
        min_keep: int = 8,
        max_keep: int | None = None,
        family: str | None = None,
        enabled: bool = True,
        profile: bool = False,
    ):
        if keep_tokens is not None:
            keep_ratio = None
        self.model = model
        self.keep_ratio = keep_ratio
        self.keep_tokens = keep_tokens
        self.min_keep = min_keep
        self.max_keep = max_keep
        self.enabled = enabled
        self.profile = profile
        self.stats: dict = {}
        self.originals: dict = {}
        if isinstance(strategy, Pruner):
            self.pruner = strategy
        else:
            if strategy not in PRUNERS:
                raise ValueError(f"unknown strategy {strategy!r}; available: {sorted(PRUNERS)}")
            cls = PRUNERS[strategy]
            self.pruner = cls(distance=distance) if strategy == "divprune" else cls()
        self.adapter = detect_adapter(model, family=family)

    def resolve_keep(self, n: int) -> int:
        return resolve_keep(
            n,
            keep_ratio=self.keep_ratio,
            keep_tokens=self.keep_tokens,
            min_keep=self.min_keep,
            max_keep=self.max_keep,
        )

    def record_stats(self, **kwargs) -> None:
        self.stats = kwargs
        if self.profile:
            logger.info("fastvision: %s", kwargs)

    def install(self) -> None:
        self.adapter.install(self.model, self)
        self.model.fastvision = self
        self.model.unwrap = self.uninstall

    def uninstall(self) -> nn.Module:
        self.adapter.uninstall(self.model)
        for attr in ("fastvision", "unwrap"):
            self.model.__dict__.pop(attr, None)
        return self.model


class FastVisionWrapper:
    """Wrap a model in one line::

        model = FastVisionWrapper(model, keep_ratio=0.1)

    Returns the same model instance with pruning installed.
    """

    def __new__(cls, model: nn.Module, **kwargs) -> nn.Module:
        if "fastvision" in model.__dict__:
            raise RuntimeError("model is already wrapped; call model.unwrap() first")
        state = FastVisionState(model, **kwargs)
        state.install()
        return model


@contextmanager
def compressed(model: nn.Module, keep_ratio: float = 0.1, **kwargs):
    """Temporarily enable pruning: ``with fastvision.compressed(model, 0.1): ...``"""
    FastVisionWrapper(model, keep_ratio=keep_ratio, **kwargs)
    try:
        yield model
    finally:
        model.unwrap()
