"""Sequence bookkeeping after visual tokens are dropped.

Pruning image features alone is not enough: the LLM scatters them into
``inputs_embeds`` at ``<image>`` placeholder positions, so the matching
placeholders must be dropped and ``attention_mask`` rebuilt to the new,
shorter length. Models with multi-dimensional rope (Qwen2-VL) additionally
need their precomputed ``position_ids`` gathered at the kept positions so
surviving visual tokens retain their original 3D coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class SplicedInputs:
    inputs_embeds: torch.Tensor  # [B, L', D]
    attention_mask: torch.Tensor  # [B, L']
    seq_len_in: int
    seq_len_out: int
    position_ids: torch.Tensor | None = None  # [P, B, L'] gathered alongside


def splice_pruned_visuals(
    input_ids: torch.Tensor,  # [B, L]
    attention_mask: torch.Tensor,  # [B, L]
    inputs_embeds: torch.Tensor,  # [B, L, D] text embeddings of input_ids
    image_token_id: int,
    pruned_features: torch.Tensor,  # [num_images, K, D] gathered by keep_index
    keep_index: torch.Tensor,  # [num_images, K] sorted, offsets within each span
    span_len: int,  # placeholder tokens per image before pruning
) -> SplicedInputs:
    """Fixed-span convenience wrapper (all images have ``span_len`` tokens)."""
    num_images = pruned_features.shape[0]
    return splice_pruned_visuals_var(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        image_token_id=image_token_id,
        pruned_features=list(pruned_features),
        keep_index=list(keep_index),
        span_lens=[span_len] * num_images,
    )


def splice_pruned_visuals_var(
    input_ids: torch.Tensor,  # [B, L]
    attention_mask: torch.Tensor,  # [B, L]
    inputs_embeds: torch.Tensor,  # [B, L, D] text embeddings of input_ids
    image_token_id: int,
    pruned_features: Sequence[torch.Tensor],  # per image [K_i, D]
    keep_index: Sequence[torch.Tensor],  # per image [K_i], offsets within the span
    span_lens: Sequence[int],  # placeholder tokens per image before pruning
    position_ids: torch.Tensor | None = None,  # [P, B, L] to gather at kept positions
) -> SplicedInputs:
    """Drop un-kept placeholder positions and scatter pruned visual features.

    Supports variable tokens-per-image (LLaVA-Next anyres, Qwen2-VL dynamic
    resolution). Images are consumed row-major across the batch, matching
    HF's ordering of ``pixel_values``. Rows are left-padded to the longest
    result (generation convention). When ``position_ids`` is given, the same
    keep mask is applied on its last dim so kept tokens retain their
    original (m-)rope coordinates.
    """
    B, L = input_ids.shape
    device = input_ids.device
    kept_embeds: list[torch.Tensor] = []
    kept_masks: list[torch.Tensor] = []
    kept_positions: list[torch.Tensor] = []
    img = 0

    for b in range(B):
        pos = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
        keep = torch.ones(L, dtype=torch.bool, device=device)
        row = inputs_embeds[b].clone()
        used = 0
        while used < pos.numel():
            if img >= len(span_lens):
                raise RuntimeError(
                    f"row {b}: more image placeholders than images "
                    f"({pos.numel() - used} left after {img} images)"
                )
            span = pos[used : used + span_lens[img]]
            if span.numel() < span_lens[img] or int(span[-1] - span[0]) != span_lens[img] - 1:
                raise RuntimeError(
                    f"row {b}, image {img}: expected a contiguous span of "
                    f"{span_lens[img]} placeholders"
                )
            kept_pos = span[keep_index[img]]
            keep[span] = False
            keep[kept_pos] = True
            row[kept_pos] = pruned_features[img].to(row.dtype)
            used += span_lens[img]
            img += 1
        kept_embeds.append(row[keep])
        kept_masks.append(attention_mask[b][keep])
        if position_ids is not None:
            kept_positions.append(position_ids[:, b, keep])

    if img != len(span_lens):
        raise RuntimeError(f"consumed {img} images but got features for {len(span_lens)}")

    out_len = max(e.shape[0] for e in kept_embeds)
    D = inputs_embeds.shape[-1]
    new_embeds = inputs_embeds.new_zeros(B, out_len, D)
    new_mask = attention_mask.new_zeros(B, out_len)
    new_positions = None
    if position_ids is not None:
        new_positions = position_ids.new_zeros(position_ids.shape[0], B, out_len)
    for b, (e, m) in enumerate(zip(kept_embeds, kept_masks)):
        new_embeds[b, out_len - e.shape[0] :] = e
        new_mask[b, out_len - m.shape[0] :] = m
        if new_positions is not None:
            new_positions[:, b, out_len - e.shape[0] :] = kept_positions[b]

    return SplicedInputs(new_embeds, new_mask, L, out_len, new_positions)
