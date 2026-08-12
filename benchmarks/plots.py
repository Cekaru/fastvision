"""Accuracy-vs-keep_ratio curves and efficiency bar charts (matplotlib)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib: pip install fastvision[bench]"
        ) from exc
    return plt


def _by_strategy(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["strategy"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: r["keep_ratio"])
    return groups


def accuracy_vs_keep(rows: list[dict], out_path: str | Path, title: str = "") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for strategy, group in _by_strategy(rows).items():
        ax.plot(
            [r["keep_ratio"] for r in group],
            [r["accuracy"] for r in group],
            marker="o",
            label=strategy,
        )
    ax.set_xlabel("keep ratio")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def strategy_comparison(rows: list[dict], out_path: str | Path, title: str = "") -> Path:
    """Grouped bars: accuracy per strategy at each keep ratio (the ablation plot)."""
    plt = _plt()
    groups = {s: g for s, g in _by_strategy(rows).items() if s != "baseline"}
    ratios = sorted({r["keep_ratio"] for g in groups.values() for r in g})
    baseline = next((r["accuracy"] for r in rows if r["strategy"] == "baseline"), None)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(groups), 1)
    for i, (strategy, group) in enumerate(sorted(groups.items())):
        acc = {r["keep_ratio"]: r["accuracy"] for r in group}
        xs = [j + i * width for j, r in enumerate(ratios) if r in acc]
        ax.bar(xs, [acc[r] for r in ratios if r in acc], width=width, label=strategy)
    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", label="baseline (no pruning)")
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(ratios))])
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_xlabel("keep ratio")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    if title:
        ax.set_title(title)
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def efficiency_bars(rows: list[dict], out_path: str | Path, title: str = "") -> Path:
    """Latency (and memory when measured) per strategy/ratio configuration."""
    plt = _plt()
    rows = sorted(rows, key=lambda r: (r["strategy"], -r["keep_ratio"]))
    labels = [f"{r['strategy']}@{r['keep_ratio']:g}" for r in rows]
    latency = [r.get("prefill_latency_s") or 0.0 for r in rows]
    has_mem = any(r.get("peak_memory_mb") for r in rows)

    fig, axes = plt.subplots(1, 2 if has_mem else 1, figsize=(10 if has_mem else 6, 4))
    ax0 = axes[0] if has_mem else axes
    ax0.bar(labels, latency)
    ax0.set_ylabel("prefill latency (s)")
    ax0.tick_params(axis="x", rotation=45)
    if has_mem:
        axes[1].bar(labels, [r.get("peak_memory_mb") or 0.0 for r in rows])
        axes[1].set_ylabel("peak memory (MB)")
        axes[1].tick_params(axis="x", rotation=45)
    if title:
        fig.suptitle(title)
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
