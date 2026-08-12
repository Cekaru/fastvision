"""Harness logic tests: no model downloads, CPU only."""

import json

import pytest
import torch

from benchmarks.datasets import Record, Task, exact_match, load_task
from benchmarks.run import evaluate_accuracy, format_markdown


def fake_task(n=4):
    records = [Record(image=None, question=f"q{i}", answers=["yes"]) for i in range(n)]
    return Task("fake", records, exact_match)


def test_evaluate_accuracy_with_fake_predict():
    task = fake_task(4)
    result = evaluate_accuracy(lambda r: "yes", task)
    assert result == {"accuracy": 1.0, "n": 4}
    result = evaluate_accuracy(lambda r: "yes" if r.question == "q0" else "no", task)
    assert result["accuracy"] == pytest.approx(0.25)


def test_synthetic_task_loads_offline():
    pytest.importorskip("PIL")
    task = load_task("synthetic", limit=3)
    assert len(task.records) == 3
    assert task.records[0].image.size == (336, 336)
    assert task.metric("noise", task.records[0].answers) == 1.0


def test_unknown_task_message():
    with pytest.raises(ValueError, match="available"):
        load_task("nope")


def test_format_markdown_and_json_roundtrip():
    rows = [
        {"strategy": "baseline", "keep_ratio": 1.0, "accuracy": 0.78, "n": 10,
         "prefill_latency_s": 0.5},
        {"strategy": "divprune", "keep_ratio": 0.1, "accuracy": 0.75, "n": 10,
         "visual_tokens_in": 576, "visual_tokens_out": 58,
         "prefill_latency_s": 0.2},
    ]
    table = format_markdown(rows)
    assert "divprune" in table and "baseline" in table
    assert "0.75" in table
    # every row renders with the same column count
    lines = table.splitlines()
    assert len({line.count("|") for line in lines}) == 1
    json.dumps(rows)  # results must be JSON-serializable


def test_measure_prefill_tiny_model():
    from benchmarks.efficiency import measure_prefill

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16)

        def forward(self, input_ids=None, **kwargs):
            out = self.head(self.emb(input_ids))
            return type("O", (), {"logits": out})()

    model = Tiny()
    inputs = {"input_ids": torch.randint(0, 16, (1, 12))}
    res = measure_prefill(model, inputs, warmup=1, iters=3)
    assert res.prefill_latency_s > 0
    assert res.seq_len == 12
    assert res.prefill_tokens_per_s > 0
    assert res.to_dict()["seq_len"] == 12


def test_plots_render(tmp_path):
    pytest.importorskip("matplotlib")
    from benchmarks.plots import accuracy_vs_keep, efficiency_bars

    rows = [
        {"strategy": "divprune", "keep_ratio": r, "accuracy": 0.7 + r / 10,
         "prefill_latency_s": r, "peak_memory_mb": 100 * r}
        for r in (0.1, 0.5, 1.0)
    ]
    p1 = accuracy_vs_keep(rows, tmp_path / "acc.png")
    p2 = efficiency_bars(rows, tmp_path / "eff.png")
    assert p1.exists() and p2.exists()
