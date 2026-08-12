# FastVision

[![CI](https://github.com/Cekaru/fastvision/actions/workflows/ci.yml/badge.svg)](https://github.com/Cekaru/fastvision/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cekaru/fastvision/blob/main/examples/llava_colab.ipynb)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Zero-fine-tuning visual token pruning for Hugging Face multimodal LLMs.
Prunes redundant visual tokens *before* they hit the language model, cutting
prefill compute and KV-cache memory — **no retraining, no architecture edits,
one line of code.**

```python
from fastvision import FastVisionWrapper

model = FastVisionWrapper(model, keep_ratio=0.1)   # that's it
out = model.generate(**inputs)                     # existing code unchanged
```

`model.unwrap()` restores the original behaviour byte-for-byte.

## Why

An MLLM turns one image into hundreds of visual tokens (LLaVA-1.5 @ 336px =
576), most of them redundant background/texture. Every one of those tokens
costs prefill FLOPs and KV-cache memory in the language model. FastVision
drops the redundant ones at the projector boundary — before the LLM — so the
whole decode runs at the shorter length, with no change to your generation
code and no fine-tuning.

## How it works

FastVision intercepts the projected visual features, selects a maximally
diverse subset with **DivPrune** (greedy Max-Min / farthest-point sampling —
training-free and attention-free, so it composes with FlashAttention/SDPA),
drops the matching placeholder positions, and rebuilds the attention mask (and
position ids for m-rope models). The KV cache follows naturally at the shorter
length. Other strategies (merging, attention-scoring) are one keyword away.

## Install

```bash
pip install git+https://github.com/Cekaru/fastvision.git
```

Requires Python ≥ 3.10, `torch` ≥ 2.1, `transformers` ≥ 4.45. For local
development use an editable install (see [Development](#development)).

## API

```python
FastVisionWrapper(
    model,
    keep_ratio=0.1,        # or keep_tokens=64
    strategy="divprune",   # "divprune" | "tome" | "fastv" | "aot" | "random" | "uniform" | "topnorm"
    distance="cosine",     # or "euclidean"
    min_keep=8,
    family=None,           # auto-detected: llava | llava_next | llava_onevision
                           #   | qwen2_vl | qwen2_5_vl | internvl
    enabled=True,          # runtime on/off toggle
)
```

- Returns the **same model object** with hooks installed — `.generate()` code
  is untouched.
- `model.unwrap()` restores original behavior exactly.
- `model.fastvision.stats` — last-run token counts and savings.
- `with fastvision.compressed(model, 0.1): ...` — temporary form.

## Supported models

| Family | Model class | Notes |
|---|---|---|
| `llava` | `LlavaForConditionalGeneration` | LLaVA-1.5; fixed 576 tokens @ 336px |
| `llava_next` | `LlavaNextForConditionalGeneration` | LLaVA-1.6 anyres: per-image variable spans |
| `llava_onevision` | `LlavaOnevisionForConditionalGeneration` | anyres per-image spans; video inputs bypass pruning |
| `qwen2_vl` | `Qwen2VLForConditionalGeneration` | dynamic resolution; kept tokens retain their 3D m-rope coordinates; video bypasses |
| `qwen2_5_vl` | `Qwen2_5_VLForConditionalGeneration` | same dynamic-resolution + m-rope contract as `qwen2_vl`; video bypasses |
| `internvl` | `InternVLForConditionalGeneration` | tiled high-res images; budgets resolved per tile |

The family is auto-detected from `model.config.model_type`; pass `family="..."`
to force it. New families are a small adapter subclass — see
[`docs/adapters.md`](docs/adapters.md).

## Strategies

| Strategy | Type | Idea |
|---|---|---|
| `divprune` | select | Max-Min diversity, farthest-point sampling (**default**) |
| `tome` | merge | bipartite soft matching, size-weighted & mass-conserving |
| `fastv` | select | text-query attention scoring at the feature boundary (experimental — attention-proxy ablation) |
| `aot` | merge | Sinkhorn optimal transport onto diverse anchors (experimental) |
| `random` / `uniform` / `topnorm` | select | ablation baselines |

Custom strategies: pass any `fastvision.pruners.Pruner` instance. Full docs
(quickstart, how it works, adapter guide, API) live in [`docs/`](docs/) and
build with `mkdocs serve` (`pip install -e .[docs]`).

## Benchmarks

One command sweeps strategies × keep ratios and writes `results.json`, a
markdown table, and accuracy/efficiency plots:

```bash
pip install "fastvision[bench] @ git+https://github.com/Cekaru/fastvision.git"
python -m benchmarks.run --model llava-hf/llava-1.5-7b-hf --task textvqa \
    --strategies divprune tome fastv random --keep-ratios 1.0 0.2 0.1 --limit 500
```

Tasks: `vqav2`, `gqa`, `textvqa` (official VQA soft accuracy / exact match),
`pope`, plus an offline `synthetic` task for smoke-testing the harness.
Efficiency numbers (prefill latency, tokens/s, peak CUDA memory) use warmup +
median-of-N. With a row cap the loaders **stream** the dataset, so only the
sampled rows are downloaded.

No GPU locally? [`examples/validate_colab.ipynb`](examples/validate_colab.ipynb)
runs the whole sweep on a real model on the Colab GPU and prints the table +
plots.

## Development

```bash
pip install -e .[dev]
ruff check fastvision benchmarks tests
pytest -q
```

Integration tests use tiny randomly initialized models (LLaVA-1.5,
LLaVA-Next, LLaVA-OneVision, Qwen2-VL, Qwen2.5-VL, InternVL) — no downloads,
runs on CPU.

## References

FastVision's strategies are independent reimplementations of published
methods (no upstream code is copied); credit to the original authors:

- **DivPrune** — Alvar, Singh, Akbari, Zhang. *DivPrune: Diversity-based
  Visual Token Pruning for Large Multimodal Models.* CVPR 2025.
  [arXiv:2503.02175](https://arxiv.org/abs/2503.02175)
- **ToMe** — Bolya, Fu, Dai, Zhang, Feichtenhofer, Hoffman. *Token Merging:
  Your ViT But Faster.* ICLR 2023.
  [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- **FastV** — Chen et al. *An Image is Worth 1/2 Tokens After Layer 2:
  Plug-and-Play Inference Acceleration for Large Vision-Language Models.*
  ECCV 2024 (Oral). [arXiv:2403.06764](https://arxiv.org/abs/2403.06764)
- **Sinkhorn / OT** (`aot` merger) — Cuturi. *Sinkhorn Distances:
  Lightspeed Computation of Optimal Transport.* NeurIPS 2013.
  [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)

## License

Apache-2.0
