"""FastV query-scoring pruner and OT merger unit tests."""

import pytest
import torch

from fastvision.pruners import FastVPruner, OTMerger


def test_fastv_keeps_query_aligned_tokens():
    D = 8
    query = torch.zeros(D)
    query[0] = 1.0
    feats = torch.randn(1, 20, D) * 0.01
    aligned = [3, 7, 15]
    for i in aligned:
        feats[0, i, 0] = 5.0  # strongly aligned with the query
    idx = FastVPruner().select(feats, 3, meta={"query": query})
    assert idx[0].tolist() == aligned


def test_fastv_fallback_without_query():
    feats = torch.randn(2, 10, 4)
    idx = FastVPruner().select(feats, 4, meta={})
    assert idx.shape == (2, 4)
    for row in idx:
        assert torch.equal(row, row.sort().values)


def test_fastv_batched_query():
    feats = torch.randn(2, 10, 4)
    q = torch.randn(2, 4)
    idx = FastVPruner().select(feats, 5, meta={"query": q})
    assert idx.shape == (2, 5)


def test_ot_merger_shapes_and_index():
    torch.manual_seed(0)
    feats = torch.randn(2, 24, 8)
    merged, idx = OTMerger().reduce(feats, 6)
    assert merged.shape == (2, 6, 8)
    assert idx.shape == (2, 6)
    for row in idx:
        assert torch.equal(row, row.sort().values)
        assert row.unique().numel() == row.numel()


def test_ot_merger_keep_all_identity():
    feats = torch.randn(1, 8, 4)
    merged, _ = OTMerger().reduce(feats, 8)
    assert torch.equal(merged, feats)


def test_ot_merged_tokens_are_finite_and_local():
    # merged outputs stay within the convex hull scale of the inputs
    torch.manual_seed(1)
    feats = torch.randn(1, 32, 8)
    merged, _ = OTMerger().reduce(feats, 4)
    assert torch.isfinite(merged).all()
    assert merged.abs().max() <= feats.abs().max() + 1e-4


def test_ot_bad_epsilon_rejected():
    with pytest.raises(ValueError, match="epsilon"):
        OTMerger(epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        OTMerger(epsilon=-0.05)


def test_ot_bad_iters_rejected():
    with pytest.raises(ValueError, match="iters"):
        OTMerger(iters=0)


def test_ot_bad_shape_rejected():
    with pytest.raises(ValueError, match="feats"):
        OTMerger().reduce(torch.randn(10, 4), 2)


def test_ot_keep_below_one_rejected():
    with pytest.raises(ValueError, match="keep"):
        OTMerger().reduce(torch.randn(1, 10, 4), 0)


def test_fastv_keep_below_one_rejected():
    with pytest.raises(ValueError, match="keep"):
        FastVPruner().select(torch.randn(1, 10, 4), keep=0)


def test_fastv_bad_query_shape_rejected():
    feats = torch.randn(2, 10, 4)
    with pytest.raises(ValueError, match="query"):
        FastVPruner().select(feats, 3, meta={"query": torch.randn(2, 8)})
    with pytest.raises(ValueError, match="query"):
        FastVPruner().select(feats, 3, meta={"query": torch.randn(3, 4)})
