"""LLaVA-1.5 adapter (``LlavaForConditionalGeneration``).

Interception strategy: on the multimodal prefill (``pixel_values`` present),
compute projected image features, run the pruner, drop the matching
``<image>`` placeholder positions, and hand the model pre-built
``inputs_embeds`` + rebuilt ``attention_mask``. ``generate()`` and the KV
cache then follow naturally at the shorter length.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..utils.bookkeeping import SplicedInputs, splice_pruned_visuals
from .base import SpliceAdapter, text_query

if TYPE_CHECKING:
    from ..wrapper import FastVisionState


def image_token_id(config) -> int:
    for attr in ("image_token_id", "image_token_index"):
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    raise RuntimeError("could not find image token id on model config")


def _stack_per_image(items) -> torch.Tensor:
    first = items[0]
    return torch.stack(list(items)) if first.dim() == 2 else torch.cat(list(items))


def _get_image_features(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    fn = getattr(model, "get_image_features", None)
    if fn is None:
        fn = model.model.get_image_features
    out = fn(pixel_values=pixel_values)
    if torch.is_tensor(out):
        feats = out
    elif hasattr(out, "pooler_output"):
        # transformers v5: projected per-image features live in pooler_output
        po = out.pooler_output
        feats = _stack_per_image(po) if isinstance(po, (list, tuple)) else po
    elif isinstance(out, (list, tuple)) and torch.is_tensor(out[0]):
        # transformers v4: tensor or list of per-image tensors
        feats = _stack_per_image(out)
    else:
        raise RuntimeError(f"unsupported get_image_features return type {type(out)}")
    if feats.dim() != 3:
        raise RuntimeError(f"unexpected image feature shape {tuple(feats.shape)}")
    return feats  # [num_images, N, D]


class LlavaAdapter(SpliceAdapter):
    name = "llava"
    consume_keys = ("pixel_values", "image_sizes")

    @classmethod
    def matches(cls, model: nn.Module) -> bool:
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        return model_type == "llava"

    def prepare(
        self,
        model: nn.Module,
        state: "FastVisionState",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kwargs: dict,
    ) -> SplicedInputs | None:
        feats = _get_image_features(model, kwargs["pixel_values"])
        n = feats.shape[1]
        k = state.resolve_keep(n)
        if k >= n:
            return None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeds = model.get_input_embeddings()(input_ids)
        img_tok = image_token_id(model.config)
        meta = {
            "family": self.name,
            "query": text_query(input_ids, embeds, attention_mask, img_tok),
        }
        pruned, keep_index = state.pruner.reduce(feats, k, meta)
        spliced = splice_pruned_visuals(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=embeds,
            image_token_id=img_tok,
            pruned_features=pruned,
            keep_index=keep_index,
            span_len=n,
        )
        state.record_stats(
            visual_tokens_in=feats.shape[0] * n,
            visual_tokens_out=feats.shape[0] * k,
            seq_len_in=spliced.seq_len_in,
            seq_len_out=spliced.seq_len_out,
        )
        return spliced
