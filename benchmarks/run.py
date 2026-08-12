"""One-command benchmark runner.

Example::

    python -m benchmarks.run --model llava-hf/llava-1.5-7b-hf --task vqav2 \
        --strategies divprune random --keep-ratios 1.0 0.5 0.2 0.1 --limit 500

Produces ``results.json``, a markdown summary table, and (with matplotlib
installed) accuracy/efficiency plots in ``--output``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch

from fastvision import FastVisionWrapper

from .datasets import Record, Task, load_task
from .efficiency import measure_prefill

# ---------------------------------------------------------------------------
# Model-agnostic evaluation (predict is injected so tests can fake it)
# ---------------------------------------------------------------------------


def evaluate_accuracy(predict: Callable[[Record], str], task: Task) -> dict:
    total = 0.0
    for record in task.records:
        total += task.metric(predict(record), record.answers)
    n = len(task.records)
    return {"accuracy": total / n if n else 0.0, "n": n}


def build_predict(
    model, processor, task: Task, max_new_tokens: int = 16
) -> Callable[[Record], str]:
    device = next(model.parameters()).device

    def predict(record: Record) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": record.image},
                    {"type": "text", "text": record.question + task.prompt_suffix},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # with pruning active generate() consumes inputs_embeds and returns
        # only new tokens; unpruned runs echo the prompt — strip it if present
        in_ids = inputs["input_ids"]
        if out.shape[-1] > in_ids.shape[-1] and torch.equal(
            out[0, : in_ids.shape[-1]], in_ids[0]
        ):
            out = out[:, in_ids.shape[-1] :]
        return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    return predict


def build_prefill_inputs(processor, task: Task, device) -> dict:
    record = task.records[0]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": record.image},
                {"type": "text", "text": record.question + task.prompt_suffix},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    model,
    processor,
    task: Task,
    strategies: list[str],
    keep_ratios: list[float],
    max_new_tokens: int = 16,
    measure_efficiency: bool = True,
    eff_iters: int = 5,
    wrap_kwargs: dict | None = None,
) -> list[dict]:
    rows: list[dict] = []
    device = next(model.parameters()).device
    prefill_inputs = build_prefill_inputs(processor, task, device) if measure_efficiency else None

    def run_one(strategy: str, ratio: float) -> dict:
        state = model.fastvision
        state.keep_ratio = ratio
        state.keep_tokens = None
        state.enabled = ratio < 1.0
        predict = build_predict(model, processor, task, max_new_tokens)
        row = {"strategy": strategy, "keep_ratio": ratio, "task": task.name}
        row.update(evaluate_accuracy(predict, task))
        row.update(state.stats)
        if prefill_inputs is not None:
            row.update(measure_prefill(model, dict(prefill_inputs), iters=eff_iters).to_dict())
        return row

    # baseline (keep_ratio >= 1.0) is strategy-independent: measure it once
    wrap_kwargs = wrap_kwargs or {}
    pruned_ratios = [r for r in keep_ratios if r < 1.0]
    if len(pruned_ratios) < len(keep_ratios):
        FastVisionWrapper(model, keep_ratio=1.0, **wrap_kwargs)
        try:
            rows.append(run_one("baseline", 1.0))
        finally:
            model.unwrap()

    for strategy in strategies:
        FastVisionWrapper(model, keep_ratio=1.0, strategy=strategy, **wrap_kwargs)
        try:
            rows.extend(run_one(strategy, ratio) for ratio in pruned_ratios)
        finally:
            model.unwrap()
    return rows


def format_markdown(rows: list[dict]) -> str:
    cols = ["strategy", "keep_ratio", "accuracy", "n",
            "visual_tokens_in", "visual_tokens_out",
            "prefill_latency_s", "peak_memory_mb"]
    present = [c for c in cols if any(c in r and r[c] is not None for r in rows)]
    lines = ["| " + " | ".join(present) + " |",
             "|" + "|".join("---" for _ in present) + "|"]
    for row in sorted(rows, key=lambda r: (r["strategy"], -r["keep_ratio"])):
        cells = []
        for c in present:
            v = row.get(c)
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v if v is not None else ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FastVision benchmark runner")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--task", default="vqav2")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=500, help="samples per config")
    parser.add_argument("--strategies", nargs="+", default=["divprune"])
    parser.add_argument("--keep-ratios", nargs="+", type=float,
                        default=[1.0, 0.5, 0.3, 0.2, 0.1, 0.05])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-keep", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    parser.add_argument("--no-efficiency", action="store_true")
    parser.add_argument("--output", default="benchmark_results")
    args = parser.parse_args(argv)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    print(f"loading {args.model} ...")
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)
    model.to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    task = load_task(args.task, split=args.split, limit=args.limit)
    print(f"task {task.name}: {len(task.records)} samples")

    rows = run_sweep(
        model, processor, task,
        strategies=args.strategies,
        keep_ratios=sorted(set(args.keep_ratios), reverse=True),
        max_new_tokens=args.max_new_tokens,
        measure_efficiency=not args.no_efficiency,
        wrap_kwargs={"min_keep": args.min_keep},
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    table = format_markdown(rows)
    (out / "results.md").write_text(table + "\n")
    print(table)

    try:
        from .plots import accuracy_vs_keep, efficiency_bars, strategy_comparison

        accuracy_vs_keep(rows, out / "accuracy_vs_keep.png", title=f"{args.model} on {task.name}")
        if len(args.strategies) > 1:
            strategy_comparison(
                rows, out / "strategy_comparison.png", title=f"{args.model} on {task.name}"
            )
        if not args.no_efficiency:
            efficiency_bars(rows, out / "efficiency.png", title=args.model)
        print(f"plots written to {out}/")
    except ImportError:
        print("matplotlib not installed; skipping plots")


if __name__ == "__main__":
    main()
