# Strategies

Select with `FastVisionWrapper(model, strategy="...")`. All are training-free.

| Strategy | Type | Idea | When to use |
|---|---|---|---|
| `divprune` | select | Max-Min diversity via farthest-point sampling | **Default.** Best accuracy retention across tasks |
| `tome` | merge | Iterative bipartite soft matching; size-weighted, mass-conserving averaging | Aggressive ratios where dropped information should be folded into survivors |
| `fastv` | select | Text-query × visual-token attention scoring at the feature boundary | Text-grounded tasks; keeps prompt-relevant tokens |
| `aot` | merge | Sinkhorn optimal transport onto farthest-point anchors (experimental) | Research/ablation |
| `random` | select | Uniform random subset | Ablation baseline |
| `uniform` | select | Even spatial stride | Ablation baseline |
| `topnorm` | select | Highest-norm tokens | Ablation baseline |

!!! note "FastV and fused attention"
    The original FastV ranks tokens by attention received inside early LLM
    layers, which requires `attn_implementation="eager"`. FastVision
    intercepts *before* the LLM, so its `fastv` scores with the same quantity
    one step earlier: the scaled dot product between the mean prompt
    embedding and each projected visual token (both live in the LLM input
    space). It therefore works with any attention implementation.

## Custom strategies

Pass any object implementing the `Pruner` interface:

```python
from fastvision.pruners import Pruner

class MyPruner(Pruner):
    def select(self, feats, keep, meta=None):   # [B, N, D] -> [B, K] sorted
        ...

model = FastVisionWrapper(model, keep_tokens=64, strategy=MyPruner())
```

Mergers additionally override `reduce(feats, keep, meta) ->
(kept_features, keep_index)` to rewrite the kept features (see
`fastvision/pruners/tome.py`).

## Budgets

- `keep_ratio=0.1` — fraction of visual tokens (per image/tile).
- `keep_tokens=64` — absolute count; takes precedence.
- `min_keep` / `max_keep` — clamps; `min_keep` defaults to 8 as a floor for
  text-heavy tasks.

Dynamic-resolution families resolve the budget **per image** (Qwen2-VL,
LLaVA-Next) or **per tile** (InternVL).

## References

Strategies are independent reimplementations of published methods — no
upstream code is copied. Credit to the original authors:

- **DivPrune** — Alvar et al., CVPR 2025.
  [arXiv:2503.02175](https://arxiv.org/abs/2503.02175)
- **ToMe** — Bolya et al., ICLR 2023.
  [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- **FastV** — Chen et al., ECCV 2024 (Oral).
  [arXiv:2403.06764](https://arxiv.org/abs/2403.06764)
- **Sinkhorn / OT** (`aot`) — Cuturi, NeurIPS 2013.
  [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)
