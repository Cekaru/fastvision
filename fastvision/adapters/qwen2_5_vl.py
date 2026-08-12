"""Qwen2.5-VL adapter (``Qwen2_5_VLForConditionalGeneration``).

Qwen2.5-VL keeps the Qwen2-VL interception contract unchanged: dynamic
resolution (per-image budgets, variable spans) and 3D m-rope coordinates
computed with the model's own ``get_rope_index`` before pruning, then
gathered at kept positions. Only the ``model_type`` and the added
video-side kwargs (``second_per_grid_ts``, ``pixel_values_videos``) differ,
so this adapter reuses :class:`Qwen2VLAdapter` wholesale and only re-declares
detection and the consume/bypass keys.
"""

from __future__ import annotations

import torch.nn as nn

from .qwen2_vl import Qwen2VLAdapter


class Qwen2_5_VLAdapter(Qwen2VLAdapter):
    name = "qwen2_5_vl"
    consume_keys = ("pixel_values", "image_grid_thw", "mm_token_type_ids")
    # video pruning is not supported yet; second_per_grid_ts only rides along
    # with video inputs, so its presence signals a video call to bypass too
    bypass_keys = ("pixel_values_videos", "second_per_grid_ts")

    @classmethod
    def matches(cls, model: nn.Module) -> bool:
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        return model_type == "qwen2_5_vl"
