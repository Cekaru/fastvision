"""InternVL adapter (``InternVLForConditionalGeneration``).

InternVL tiles high-resolution images and every tile contributes a fixed
number of projected tokens (after pixel-shuffle downsampling), scattered into
one contiguous placeholder span per image. That is exactly the LLaVA
fixed-span layout with tiles in place of images, so the LLaVA prepare logic
applies unchanged — budgets are resolved per tile.
"""

from __future__ import annotations

import torch.nn as nn

from .llava import LlavaAdapter


class InternVLAdapter(LlavaAdapter):
    name = "internvl"
    consume_keys = ("pixel_values",)

    @classmethod
    def matches(cls, model: nn.Module) -> bool:
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        return model_type == "internvl"
