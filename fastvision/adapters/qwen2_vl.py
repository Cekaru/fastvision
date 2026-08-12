"""Qwen2-VL adapter (``Qwen2VLForConditionalGeneration``).

Two things make this family the generality test:

- **Dynamic resolution**: each image contributes ``prod(grid_thw) /
  spatial_merge_size**2`` tokens, so budgets are resolved per image and the
  splice handles variable span lengths.
- **M-RoPE**: vision tokens carry 3D (temporal/height/width) rope
  coordinates. We compute the full-sequence 3D ``position_ids`` with the
  model's own ``get_rope_index`` *before* pruning, then gather the kept
  positions, so surviving tokens keep their original spatial coordinates.
  A text-only position row is prepended (the ``[4, B, L]`` convention the
  text model expects) and the positions are handed to ``generate``, which
  slices them per step and extends by +1 per dim for decoded text tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..utils.bookkeeping import SplicedInputs, splice_pruned_visuals_var
from .base import SpliceAdapter, text_query

if TYPE_CHECKING:
    from ..wrapper import FastVisionState


def _tokens_per_image(model: nn.Module, image_grid_thw: torch.Tensor) -> list[int]:
    merge = model.config.vision_config.spatial_merge_size
    return (image_grid_thw.prod(-1) // merge**2).tolist()


def _get_image_features_list(
    model: nn.Module, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
) -> list[torch.Tensor]:
    out = model.get_image_features(pixel_values, image_grid_thw)
    if hasattr(out, "pooler_output"):  # transformers v5
        out = out.pooler_output
    if torch.is_tensor(out):  # transformers v4: concatenated [total, D]
        out = torch.split(out, _tokens_per_image(model, image_grid_thw))
    if not isinstance(out, (list, tuple)) or not torch.is_tensor(out[0]):
        raise RuntimeError(f"unsupported get_image_features return type {type(out)}")
    return list(out)  # num_images x [N_i, D]


def _get_rope_index(
    model: nn.Module,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Full-sequence 3D position ids ``[3, B, L]`` for the unpruned input."""
    if mm_token_type_ids is None:
        mm_token_type_ids = (input_ids == model.config.image_token_id).to(torch.int)
    try:  # transformers v5
        position_ids, _ = model.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
    except TypeError:  # transformers v4: no mm_token_type_ids argument
        position_ids, _ = model.model.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )
    return position_ids


class Qwen2VLAdapter(SpliceAdapter):
    name = "qwen2_vl"
    consume_keys = ("pixel_values", "image_grid_thw", "mm_token_type_ids")
    bypass_keys = ("pixel_values_videos",)  # video pruning not supported yet

    @classmethod
    def matches(cls, model: nn.Module) -> bool:
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        return model_type == "qwen2_vl"

    def prepare(
        self,
        model: nn.Module,
        state: "FastVisionState",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kwargs: dict,
    ) -> SplicedInputs | None:
        image_grid_thw = kwargs.get("image_grid_thw")
        if image_grid_thw is None:
            raise RuntimeError("qwen2_vl pruning needs image_grid_thw alongside pixel_values")
        feats = _get_image_features_list(model, kwargs["pixel_values"], image_grid_thw)

        span_lens = [f.shape[0] for f in feats]
        keeps = [state.resolve_keep(n) for n in span_lens]
        if all(k >= n for k, n in zip(keeps, span_lens)):
            return None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeds = model.get_input_embeddings()(input_ids)
        meta = {
            "family": self.name,
            "query": text_query(
                input_ids, embeds, attention_mask, model.config.image_token_id
            ),
        }
        keep_index, pruned = [], []
        for f, k in zip(feats, keeps):
            p, idx = state.pruner.reduce(f.unsqueeze(0), k, meta)
            keep_index.append(idx[0])
            pruned.append(p[0])

        position_ids = _get_rope_index(
            model, input_ids, image_grid_thw, attention_mask,
            kwargs.get("mm_token_type_ids"),
        )
        spliced = splice_pruned_visuals_var(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=embeds,
            image_token_id=model.config.image_token_id,
            pruned_features=pruned,
            keep_index=keep_index,
            span_lens=span_lens,
            position_ids=position_ids,
        )
        # prepend the text-only row ([4, B, L'] packed convention) so the
        # text model builds its causal mask from 1D positions
        text_pos = (spliced.attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        spliced.position_ids = torch.cat([text_pos.unsqueeze(0), spliced.position_ids])
        # stale prefill deltas from a previous unpruned run must not leak in
        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = None

        state.record_stats(
            visual_tokens_in=sum(span_lens),
            visual_tokens_out=sum(k.numel() for k in keep_index),
            seq_len_in=spliced.seq_len_in,
            seq_len_out=spliced.seq_len_out,
        )
        return spliced
