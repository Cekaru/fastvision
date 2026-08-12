import pytest
import torch

from fastvision.pruners import RandomPruner, TopNormPruner, UniformPruner, resolve_keep


def test_ratio():
    assert resolve_keep(576, keep_ratio=0.1) == 58
    assert resolve_keep(576, keep_ratio=1.0) == 576


def test_absolute_overrides_ratio():
    assert resolve_keep(576, keep_ratio=0.1, keep_tokens=64) == 64


def test_min_max_clamps():
    assert resolve_keep(576, keep_ratio=0.001, min_keep=8) == 8
    assert resolve_keep(576, keep_ratio=0.9, max_keep=100) == 100


def test_never_exceeds_n():
    assert resolve_keep(10, keep_tokens=500) == 10
    assert resolve_keep(4, keep_ratio=0.5, min_keep=8) == 4


def test_invalid_inputs():
    with pytest.raises(ValueError):
        resolve_keep(100)
    with pytest.raises(ValueError):
        resolve_keep(100, keep_ratio=0.0)
    with pytest.raises(ValueError):
        resolve_keep(100, keep_ratio=1.5)


@pytest.mark.parametrize("cls", [RandomPruner, UniformPruner, TopNormPruner])
def test_baseline_contract(cls):
    feats = torch.randn(2, 30, 8, generator=torch.Generator().manual_seed(0))
    idx = cls().select(feats, keep=6)
    assert idx.shape == (2, 6)
    assert idx.dtype == torch.long
    for row in idx:
        assert torch.equal(row, row.sort().values)
        assert row.unique().numel() == 6
    assert torch.equal(idx, cls().select(feats, keep=6))
