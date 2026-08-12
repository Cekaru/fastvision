# Writing an adapter

An adapter knows two things about a model family: **where the projected
visual features live** and **how to fix up the sequence** after tokens are
dropped. Everything else (hook install/uninstall, skip conditions, stats) is
shared machinery.

## The shape of the problem

After the projector, visual features are scattered into `inputs_embeds` at
`<image>` placeholder positions. Dropping features therefore also requires
dropping placeholders and rebuilding `attention_mask` (and `position_ids`
for multi-dimensional rope). FastVision does this by handing the model
pre-built `inputs_embeds` for the prefill.

## Subclass `SpliceAdapter`

```python
from fastvision.adapters.base import SpliceAdapter, text_query
from fastvision.utils.bookkeeping import splice_pruned_visuals


class MyFamilyAdapter(SpliceAdapter):
    name = "my_family"
    consume_keys = ("pixel_values",)   # kwargs replaced by the splice
    bypass_keys = ()                   # kwargs that force a bypass (e.g. video)

    @classmethod
    def matches(cls, model):
        return getattr(model.config, "model_type", "") == "my_family"

    def prepare(self, model, state, input_ids, attention_mask, kwargs):
        feats = ...                    # [num_images, N, D] projected features
        k = state.resolve_keep(feats.shape[1])
        if k >= feats.shape[1]:
            return None                # pruning is a no-op

        embeds = model.get_input_embeddings()(input_ids)
        meta = {"family": self.name,
                "query": text_query(input_ids, embeds, attention_mask, img_tok)}
        pruned, keep_index = state.pruner.reduce(feats, k, meta)

        spliced = splice_pruned_visuals(
            input_ids=input_ids, attention_mask=attention_mask,
            inputs_embeds=embeds, image_token_id=img_tok,
            pruned_features=pruned, keep_index=keep_index,
            span_len=feats.shape[1],
        )
        state.record_stats(...)
        return spliced
```

Register it:

```python
from fastvision.adapters.registry import ADAPTERS
ADAPTERS["my_family"] = MyFamilyAdapter
```

## Variable spans and m-rope

- Families with variable tokens per image (anyres, dynamic resolution) use
  `splice_pruned_visuals_var` with per-image `span_lens` and per-image
  budgets — see `llava_next.py`.
- Families with multi-dimensional rope must compute full-sequence
  `position_ids` *before* pruning and pass them to the splice, which gathers
  them at kept positions — see `qwen2_vl.py`.

## Rules

1. **Symmetry.** After `uninstall()` the model must behave byte-identically.
   The tiny-model tests assert `torch.equal` on logits.
2. **Fail loudly.** If required kwargs are missing or shapes are unexpected,
   raise with an actionable message; never silently mis-prune.
3. **Bypass what you can't handle.** Video inputs, cached decode steps and
   `labels` (training) fall through to the original forward.
4. **Add a tiny-model integration test.** Randomly initialized config, no
   downloads, CPU-friendly — see `tests/integration/`.
