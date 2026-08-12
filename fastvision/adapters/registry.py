"""Adapter detection and registry."""

from __future__ import annotations

import torch.nn as nn

from .base import Adapter
from .internvl import InternVLAdapter
from .llava import LlavaAdapter
from .llava_next import LlavaNextAdapter
from .llava_onevision import LlavaOnevisionAdapter
from .qwen2_5_vl import Qwen2_5_VLAdapter
from .qwen2_vl import Qwen2VLAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    "llava": LlavaAdapter,
    "llava_next": LlavaNextAdapter,
    "llava_onevision": LlavaOnevisionAdapter,
    "qwen2_vl": Qwen2VLAdapter,
    "qwen2_5_vl": Qwen2_5_VLAdapter,
    "internvl": InternVLAdapter,
}


def detect_adapter(model: nn.Module, family: str | None = None) -> Adapter:
    if family is not None:
        if family not in ADAPTERS:
            raise ValueError(
                f"unknown family {family!r}; supported: {sorted(ADAPTERS)}"
            )
        return ADAPTERS[family]()
    for cls in ADAPTERS.values():
        if cls.matches(model):
            return cls()
    model_type = getattr(getattr(model, "config", None), "model_type", type(model).__name__)
    raise ValueError(
        f"no adapter for model type {model_type!r}. Supported families: "
        f"{sorted(ADAPTERS)}. Pass family=... to override, or subclass "
        "fastvision.adapters.Adapter to add support."
    )
