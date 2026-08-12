# API reference

## `FastVisionWrapper`

```python
FastVisionWrapper(
    model,                          # HF multimodal model (in-place)
    keep_ratio: float = 0.1,        # fraction of visual tokens to keep
    keep_tokens: int | None = None, # absolute budget; overrides keep_ratio
    strategy: str | Pruner = "divprune",
    distance: str = "cosine",       # divprune only: "cosine" | "euclidean"
    min_keep: int = 8,              # floor per image/tile
    max_keep: int | None = None,    # ceiling per image/tile
    family: str | None = None,      # auto-detected; override to force
    enabled: bool = True,           # runtime on/off toggle
    profile: bool = False,          # log token counts per run
) -> nn.Module                       # the same model object, hooks installed
```

Strategies: `divprune`, `tome`, `fastv`, `aot`, `random`, `uniform`,
`topnorm` — or any `Pruner` instance.

### Installed attributes

| Attribute | Meaning |
|---|---|
| `model.fastvision` | the `FastVisionState` (config + stats) |
| `model.fastvision.stats` | last-run dict: `visual_tokens_in/out`, `seq_len_in/out` |
| `model.fastvision.enabled` | toggle pruning at runtime |
| `model.fastvision.keep_ratio` / `.keep_tokens` | adjustable between calls |
| `model.unwrap()` | remove hooks, restore byte-identical behavior |

## `fastvision.compressed`

```python
with fastvision.compressed(model, keep_ratio=0.1, **wrapper_kwargs):
    model.generate(**inputs)
# automatically unwrapped
```

## `fastvision.pruners`

```python
class Pruner(ABC):
    def select(self, feats, keep, meta=None) -> Tensor:
        """[B, N, D] -> keep_index [B, K], sorted ascending per row."""

    def reduce(self, feats, keep, meta=None) -> tuple[Tensor, Tensor]:
        """(kept_features [B, K, D], keep_index [B, K]).
        Default gathers at select(); mergers override."""
```

`meta` keys provided by adapters: `family` (adapter name), `query`
(mean prompt embedding `[D]`, used by `fastv`).

```python
resolve_keep(n, keep_ratio=None, keep_tokens=None, min_keep=1, max_keep=None) -> int
```

## `fastvision.adapters`

- `Adapter` / `SpliceAdapter` — base classes ([guide](adapters.md))
- `ADAPTERS` — registry dict, extensible at runtime
- `detect_adapter(model, family=None)` — resolution logic

## Behavior notes

- Pruning applies to the multimodal **prefill** only; decode steps and calls
  with `past_key_values` pass through untouched.
- Calls with `labels` (training) bypass pruning with a warning.
- Video inputs (`pixel_values_videos`) bypass pruning (Qwen2-VL).
- With pruning active, `generate()` consumes `inputs_embeds`, so the returned
  ids contain **only new tokens** (no prompt echo).
