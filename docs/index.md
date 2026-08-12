# FastVision in 60 seconds

FastVision prunes redundant visual tokens *before* they hit the language
model — cutting prefill compute and KV-cache memory with **zero fine-tuning**
and **three lines of code**.

## Install

```bash
pip install fastvision            # core: torch + transformers
pip install fastvision[bench]     # + benchmark harness deps
```

## Use

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from fastvision import FastVisionWrapper

model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = FastVisionWrapper(model, keep_ratio=0.1)   # that's it

out = model.generate(**inputs)                     # existing code unchanged
```

The wrapper returns the **same model object** with hooks installed, so
nothing downstream changes. To undo:

```python
model.unwrap()          # restores original behavior exactly
```

Or temporarily:

```python
import fastvision
with fastvision.compressed(model, keep_ratio=0.1):
    out = model.generate(**inputs)
```

## Inspect what happened

```python
model.fastvision.stats
# {'visual_tokens_in': 576, 'visual_tokens_out': 58,
#  'seq_len_in': 610, 'seq_len_out': 92}
```

## Supported families

| Family | Model class | Notes |
|---|---|---|
| `llava` | `LlavaForConditionalGeneration` | fixed 576 tokens @ 336px |
| `llava_next` | `LlavaNextForConditionalGeneration` | anyres, per-image variable spans |
| `llava_onevision` | `LlavaOnevisionForConditionalGeneration` | anyres per-image spans; video bypasses |
| `qwen2_vl` | `Qwen2VLForConditionalGeneration` | dynamic resolution + m-rope |
| `qwen2_5_vl` | `Qwen2_5_VLForConditionalGeneration` | dynamic resolution + m-rope; video bypasses |
| `internvl` | `InternVLForConditionalGeneration` | tiled high-res, per-tile budgets |

The family is auto-detected from `model.config.model_type`; pass
`family="..."` to override. Unknown models raise a clear error — see
[Writing an adapter](adapters.md).
