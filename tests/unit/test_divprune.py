import torch

from fastvision.pruners import DivPrune


def make_clusters(centers: torch.Tensor, per_cluster: int, noise: float = 0.01):
    """[C, D] centers -> [1, C*per_cluster, D] points, cluster id per token."""
    gen = torch.Generator().manual_seed(0)
    points, labels = [], []
    for c, center in enumerate(centers):
        pts = center.unsqueeze(0) + noise * torch.randn(
            per_cluster, centers.shape[1], generator=gen
        )
        points.append(pts)
        labels += [c] * per_cluster
    return torch.cat(points).unsqueeze(0), torch.tensor(labels)


def orthogonal_centers(c: int, d: int, scale: float = 5.0) -> torch.Tensor:
    eye = torch.eye(d)[:c]
    return eye * scale


def test_one_token_per_cluster():
    centers = orthogonal_centers(4, 16)
    feats, labels = make_clusters(centers, per_cluster=20)
    idx = DivPrune().select(feats, keep=4)
    picked = labels[idx[0]]
    assert sorted(picked.tolist()) == [0, 1, 2, 3]


def test_euclidean_covers_clusters():
    centers = orthogonal_centers(3, 8)
    feats, labels = make_clusters(centers, per_cluster=10)
    idx = DivPrune(distance="euclidean").select(feats, keep=3)
    assert sorted(labels[idx[0]].tolist()) == [0, 1, 2]


def test_shapes_dtype_sorted_unique():
    gen = torch.Generator().manual_seed(1)
    feats = torch.randn(3, 50, 32, generator=gen)
    idx = DivPrune().select(feats, keep=10)
    assert idx.shape == (3, 10)
    assert idx.dtype == torch.long
    for row in idx:
        assert torch.equal(row, row.sort().values)
        assert row.unique().numel() == 10
        assert row.min() >= 0 and row.max() < 50


def test_deterministic():
    gen = torch.Generator().manual_seed(2)
    feats = torch.randn(2, 40, 16, generator=gen)
    p = DivPrune()
    assert torch.equal(p.select(feats, keep=8), p.select(feats, keep=8))


def test_keep_clamped_to_n():
    feats = torch.randn(1, 5, 4)
    idx = DivPrune().select(feats, keep=99)
    assert idx.shape == (1, 5)
    assert sorted(idx[0].tolist()) == [0, 1, 2, 3, 4]


def test_batch_rows_independent():
    centers_a = orthogonal_centers(2, 8)
    centers_b = -orthogonal_centers(2, 8)
    a, la = make_clusters(centers_a, per_cluster=10)
    b, lb = make_clusters(centers_b, per_cluster=10)
    feats = torch.cat([a, b])
    idx = DivPrune().select(feats, keep=2)
    assert sorted(la[idx[0]].tolist()) == [0, 1]
    assert sorted(lb[idx[1]].tolist()) == [0, 1]


def test_bad_distance_rejected():
    import pytest

    with pytest.raises(ValueError):
        DivPrune(distance="manhattan")


def test_keep_below_one_rejected():
    import pytest

    with pytest.raises(ValueError, match="keep"):
        DivPrune().select(torch.randn(1, 10, 4), keep=0)
