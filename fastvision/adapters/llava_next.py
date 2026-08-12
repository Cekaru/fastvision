"""LLaVA-1.6 / LLaVA-Next adapter (``LlavaNextForConditionalGeneration``).

Anyres models produce a *variable* number of visual tokens per image (base
tile + high-res patches + newline separators), so pruning budgets are
resolved per image and the splice handles variable span lengths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..utils.bookkeeping import SplicedInputs, splice_pruned_visuals_var
from .base import SpliceAdapter, text_query
from .llava import image_token_id

if TYPE_CHECKING:
    from ..wrapper import FastVisionState


def _get_image_features_list(
    model: nn.Module, pixel_values: torch.Tensor, image_sizes: torch.Tensor
) -> list[torch.Tensor]:
    """Per-image packed features (variable length, newlines included)."""
    fn = getattr(model, "get_image_features", None)
    if fn is None:
        fn = model.model.get_image_features
    out = fn(pixel_values=pixel_values, image_sizes=image_sizes)
    if hasattr(out, "pooler_output"):  # transformers v5
        out = out.pooler_output
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], (list, tuple)):
        out = out[0]  # transformers v4: (features, feature_lens)
    if not isinstance(out, (list, tuple)) or not torch.is_tensor(out[0]):
        raise RuntimeError(f"unsupported get_image_features return type {type(out)}")
    return list(out)  # num_images x [N_i, D]


class LlavaNextAdapter(SpliceAdapter):
    name = "llava_next"
    consume_keys = ("pixel_values", "image_sizes")

    @classmethod
    def matches(cls, model: nn.Module) -> bool:
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        return model_type == "llava_next"

    def prepare(
        self,
        model: nn.Module,
        state: "FastVisionState",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kwargs: dict,
    ) -> SplicedInputs | None:
        image_sizes = kwargs.get("image_sizes")
        if image_sizes is None:
            raise RuntimeError("llava_next pruning needs image_sizes alongside pixel_values")
        feats = _get_image_features_list(model, kwargs["pixel_values"], image_sizes)

        # per-image budget: anyres token counts differ between images
        span_lens = [f.shape[0] for f in feats]
        keeps = [state.resolve_keep(n) for n in span_lens]
        if all(k >= n for k, n in zip(keeps, span_lens)):
            return None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeds = model.get_input_embeddings()(input_ids)
        img_tok = image_token_id(model.config)
        meta = {
            "family": self.name,
            "query": text_query(input_ids, embeds, attention_mask, img_tok),
        }
        keep_index, pruned = [], []
        for f, k in zip(feats, keeps):
            p, idx = state.pruner.reduce(f.unsqueeze(0), k, meta)
            keep_index.append(idx[0])
            pruned.append(p[0])

        spliced = splice_pruned_visuals_var(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=embeds,
            image_token_id=img_tok,
            pruned_features=pruned,
            keep_index=keep_index,
            span_lens=span_lens,
        )
        state.record_stats(
            visual_tokens_in=sum(span_lens),
            visual_tokens_out=sum(k.numel() for k in keep_index),
            seq_len_in=spliced.seq_len_in,
            seq_len_out=spliced.seq_len_out,
        )
        return spliced
