# Benchmarks

One command sweeps strategies × keep ratios, evaluates accuracy and
efficiency, and writes plots:

```bash
pip install fastvision[bench]

python -m benchmarks.run \
    --model llava-hf/llava-1.5-7b-hf \
    --task vqav2 \
    --strategies divprune tome fastv random \
    --keep-ratios 1.0 0.5 0.3 0.2 0.1 0.05 \
    --limit 500
```

Outputs in `benchmark_results/`:

- `results.json` / `results.md` — raw rows and a markdown table
- `accuracy_vs_keep.png` — accuracy-vs-keep-ratio curve per strategy
- `strategy_comparison.png` — grouped-bar ablation (with the unpruned
  baseline as a dashed line)
- `efficiency.png` — prefill latency and peak CUDA memory per configuration

## Tasks

| Task | Metric | Notes |
|---|---|---|
| `vqav2` | official VQA soft accuracy | answer normalization per the VQA eval spec |
| `gqa` | exact match | |
| `textvqa` | VQA soft accuracy | text-heavy → most pruning-sensitive |
| `pope` | exact match | hallucination probe |
| `synthetic` | exact match | offline, for smoke-testing the harness |

## Methodology

- The `keep_ratio=1.0` baseline is strategy-independent and measured once.
- Efficiency numbers (prefill latency, tokens/s, peak CUDA memory) use CUDA
  events with warmup and median-of-N (default N=5).
- Each configuration wraps and unwraps the model, so runs are independent.

## Reproducing the README table

```bash
python -m benchmarks.run --model llava-hf/llava-1.5-7b-hf --task vqav2 \
    --strategies divprune random --keep-ratios 1.0 0.2 0.1 --limit 1000
python -m benchmarks.run --model Qwen/Qwen2-VL-7B-Instruct --task textvqa \
    --strategies divprune tome --keep-ratios 1.0 0.2 0.1 --limit 1000
```
