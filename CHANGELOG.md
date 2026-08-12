# Changelog

All notable changes to FastVision are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- One-line `FastVisionWrapper` around Hugging Face multimodal LLMs:
  training-free visual-token pruning at the projector boundary, with
  `unwrap()` for exact restoration and a `compressed()` context manager.
  `generate()`/`forward()` are intercepted transparently and the attention
  mask, position ids, and KV cache follow at the shorter length.
- Pruning strategies behind `strategy=`: DivPrune (default, diversity-based
  farthest-point selection), a ToMe merger (size-weighted, mass-conserving),
  FastV-style attention scoring at the feature boundary, an experimental
  Sinkhorn optimal-transport merger, and random / uniform / top-norm
  baselines. Custom strategies via the `Pruner` interface.
- Model families, auto-detected from `model.config.model_type`: LLaVA-1.5,
  LLaVA-Next, LLaVA-OneVision, Qwen2-VL, Qwen2.5-VL, and InternVL. Dynamic
  resolution and m-rope models keep per-image budgets and gather 3D positions
  at surviving tokens; video inputs bypass pruning.
- Budget control via `keep_ratio` / `keep_tokens` with `min_keep` / `max_keep`
  clamps, resolved per image or per tile for dynamic-resolution families.
- Benchmark harness (`python -m benchmarks.run`): VQAv2 / GQA / TextVQA / POPE
  with official VQA answer normalization and soft accuracy, an offline
  `synthetic` task, prefill-latency / tokens-per-second / peak-memory
  measurement (warmup + median-of-N), strategy × keep-ratio sweeps, and
  accuracy/efficiency plots. Dataset loaders stream when a row cap is set, so
  only the sampled rows are downloaded.
- Documentation site (mkdocs-material), Colab notebooks, and a token-selection
  visualizer.

[Unreleased]: https://github.com/Cekaru/fastvision
