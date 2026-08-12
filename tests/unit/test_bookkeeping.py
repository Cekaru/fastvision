"""Variable-span splicing and position-id gathering."""

import pytest
import torch

from fastvision.utils.bookkeeping import (
    splice_pruned_visuals,
    splice_pruned_visuals_var,
)

IMG = 99
D = 4


def build_row(*segments):
    return torch.tensor([t for seg in segments for t in seg])


def test_variable_spans_and_position_gather():
    # row 0: span of 4 at positions 2..5; row 1: span of 2 at positions 5..6
    ids0 = build_row([1, 2], [IMG] * 4, [3, 4, 5, 6])
    ids1 = build_row([1, 2, 3, 4, 5], [IMG] * 2, [6, 7, 8])
    input_ids = torch.stack([ids0, ids1])
    B, L = input_ids.shape
    attention_mask = torch.ones(B, L, dtype=torch.long)
    embeds = torch.randn(B, L, D)
    position_ids = torch.arange(L).view(1, 1, L).expand(3, B, L).clone()
    position_ids[1] += 100  # distinguish dims

    feats = [torch.full((2, D), 10.0), torch.full((1, D), 20.0)]
    keep = [torch.tensor([0, 3]), torch.tensor([1])]

    out = splice_pruned_visuals_var(
        input_ids, attention_mask, embeds, IMG,
        pruned_features=feats, keep_index=keep, span_lens=[4, 2],
        position_ids=position_ids,
    )
    assert out.seq_len_in == 10
    assert out.seq_len_out == 9  # row 0 drops 2, row 1 drops 1 -> padded to 9
    # row 0 is left-padded by 1; kept visual tokens carry pruned features
    assert out.attention_mask[0].tolist() == [0] + [1] * 8
    assert torch.allclose(out.inputs_embeds[0, 3], torch.full((D,), 10.0))
    assert torch.allclose(out.inputs_embeds[0, 4], torch.full((D,), 10.0))
    # row 0 kept positions: 0,1 (text), 2 and 5 (visual), 6..9 (text)
    assert out.position_ids[0, 0, 1:].tolist() == [0, 1, 2, 5, 6, 7, 8, 9]
    assert out.position_ids[1, 0, 1:].tolist() == [100, 101, 102, 105, 106, 107, 108, 109]
    # row 1 keeps offset 1 of its span (position 6), no padding
    assert out.attention_mask[1].tolist() == [1] * 9
    assert out.position_ids[0, 1].tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    assert torch.allclose(out.inputs_embeds[1, 5], torch.full((D,), 20.0))


def test_fixed_span_wrapper_matches_var():
    ids = build_row([1], [IMG] * 4, [2, 3])
    input_ids = ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    embeds = torch.randn(1, input_ids.shape[1], D)
    feats = torch.randn(1, 2, D)
    keep = torch.tensor([[1, 2]])

    fixed = splice_pruned_visuals(input_ids, attention_mask, embeds, IMG, feats, keep, 4)
    var = splice_pruned_visuals_var(
        input_ids, attention_mask, embeds, IMG, [feats[0]], [keep[0]], [4]
    )
    assert torch.equal(fixed.inputs_embeds, var.inputs_embeds)
    assert torch.equal(fixed.attention_mask, var.attention_mask)


def test_adjacent_images_in_one_run():
    # two images back-to-back with no separator: 3 + 2 placeholders
    ids = build_row([1], [IMG] * 5, [2])
    input_ids = ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    embeds = torch.zeros(1, 7, D)
    feats = [torch.full((1, D), 1.0), torch.full((1, D), 2.0)]
    keep = [torch.tensor([2]), torch.tensor([0])]

    out = splice_pruned_visuals_var(
        input_ids, attention_mask, embeds, IMG, feats, keep, span_lens=[3, 2]
    )
    assert out.seq_len_out == 4
    assert torch.allclose(out.inputs_embeds[0, 1], torch.full((D,), 1.0))  # img 0, offset 2
    assert torch.allclose(out.inputs_embeds[0, 2], torch.full((D,), 2.0))  # img 1, offset 0


def test_span_mismatch_raises():
    ids = build_row([1], [IMG] * 3, [2])
    input_ids = ids.unsqueeze(0)
    args = (torch.ones_like(input_ids), torch.zeros(1, 5, D), IMG)
    with pytest.raises(RuntimeError, match="contiguous span"):
        splice_pruned_visuals_var(
            input_ids, *args,
            pruned_features=[torch.zeros(1, D)], keep_index=[torch.tensor([0])],
            span_lens=[4],
        )
    with pytest.raises(RuntimeError, match="more image placeholders"):
        splice_pruned_visuals_var(
            input_ids, *args,
            pruned_features=[torch.zeros(1, D)], keep_index=[torch.tensor([0])],
            span_lens=[2],
        )
