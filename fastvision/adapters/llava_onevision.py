"""LLaVA-OneVision adapter (``LlavaOnevisionForConditionalGeneration``).

LLaVA-OneVision is an anyres model like LLaVA-Next: each image expands to a
variable number of visual tokens (base tile + high-res patches + newline
separators), so budgets are resolved per image and the splice handles
variable span lengths. Image features are extracted with ``image_sizes``,
exactly as in LLaVA-Next. Video inputs (``pixel_values_videos``) bypass
pruning.
"""

from __future__ import annotations

import torch.nn as nn

from .llava_next import LlavaNextAdapter


class LlavaOnevisionAdapter(LlavaNextAdapter):
    name = "llava_onevision"
    consume_keys = ("pixel_values", "image_sizes")
    bypass_keys = ("pixel_values_videos",)  # video pruning not supported yet

    @classmethod
    def matches(cls, model: nn.Module) -> bool:
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        return model_type == "llava_onevision"
