"""ToMe merger: shapes, index contract, mass conservation, merging behavior."""

import pytest
import torch

from fastvision.pruners import ToMeMerger


def test_shapes_and_index_contract():
    torch.manual_seed(0)
    feats = torch.randn(3, 20, 8)
    merged, idx = ToMeMerger().reduce(feats, 7)
    assert merged.shape == (3, 7, 8)
    assert idx.shape == (3, 7)
    assert idx.dtype == torch.long
    for row in idx:
        assert torch.equal(row, row.sort().values)
        assert row.unique().numel() == row.numel()
        assert row.min() >= 0 and row.max() < 20


def test_keep_all_is_identity():
    feats = torch.randn(2, 10, 4)
    merged, idx = ToMeMerger().reduce(feats, 10)
    assert torch.equal(merged, feats)
    assert torch.equal(idx[0], torch.arange(10))


def test_full_merge_is_global_mean():
    # size-weighted merging all the way to one token must conserve mass
    torch.manual_seed(1)
    feats = torch.randn(2, 16, 6)
    merged, idx = ToMeMerger().reduce(feats, 1)
    assert merged.shape == (2, 1, 6)
    torch.testing.assert_close(merged[:, 0], feats.mean(dim=1), atol=1e-5, rtol=1e-5)


def test_duplicates_merge_first():
    # two identical tokens should merge together, keeping the distinct ones
    base = torch.eye(4).repeat_interleave(2, dim=0).unsqueeze(0)  # 8 tokens, 4 directions
    merged, idx = ToMeMerger().reduce(base, 4)
    # each survivor equals one of the four distinct directions
    for k in range(4):
        assert (merged[0] - torch.eye(4)[k]).norm(dim=-1).min() < 1e-5


def test_deterministic():
    feats = torch.randn(1, 30, 8)
    a = ToMeMerger().reduce(feats, 5)
    b = ToMeMerger().reduce(feats, 5)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_select_matches_reduce_index():
    feats = torch.randn(2, 12, 4)
    m = ToMeMerger()
    assert torch.equal(m.select(feats, 5), m.reduce(feats, 5)[1])


def test_bad_shape_rejected():
    with pytest.raises(ValueError):
        ToMeMerger().reduce(torch.randn(5, 4), 2)


def test_keep_below_one_rejected():
    with pytest.raises(ValueError, match="keep"):
        ToMeMerger().reduce(torch.randn(1, 10, 4), 0)
