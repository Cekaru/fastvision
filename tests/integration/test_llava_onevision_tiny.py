"""End-to-end test on a tiny randomly initialized LLaVA-OneVision (no downloads).

Anyres like LLaVA-Next: variable token count per image. Guards that the thin
LLaVA-OneVision adapter detects and prunes with per-image budgets, and that
video inputs bypass pruning.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import (  # noqa: E402
    LlavaOnevisionConfig,
    LlavaOnevisionForConditionalGeneration,
)

from fastvision import FastVisionWrapper  # noqa: E402

TILE = 16  # vision tower input size
PATCH = 8
VOCAB = 128
IMAGE_TOKEN = VOCAB - 1
PINPOINTS = [[16, 16], [16, 32], [32, 16], [32, 32]]


@pytest.fixture()
def model():
    cfg = LlavaOnevisionConfig(
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": TILE,
            "patch_size": PATCH,
        },
        text_config={
            "model_type": "qwen2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": VOCAB,
            "max_position_embeddings": 256,
        },
        image_token_index=IMAGE_TOKEN,
        image_grid_pinpoints=PINPOINTS,
    )
    torch.manual_seed(0)
    m = LlavaOnevisionForConditionalGeneration(cfg)
    m.eval()
    return m


def make_inputs(model, image_sizes=((16, 16),)):
    """Build inputs whose placeholder spans match the model's real feature lens."""
    from transformers.models.llava_onevision.modeling_llava_onevision import (
        image_size_to_num_patches,
    )

    torch.manual_seed(1)
    sizes = torch.tensor(list(image_sizes), dtype=torch.long)
    num_patches = [
        image_size_to_num_patches(s, PINPOINTS, TILE) for s in sizes.tolist()
    ]
    pixel_values = torch.randn(len(sizes), max(num_patches), 3, TILE, TILE)

    with torch.no_grad():
        feats = model.get_image_features(pixel_values=pixel_values, image_sizes=sizes)
    feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
    span_lens = [f.shape[0] for f in feats]

    parts = [torch.tensor([1, 5, 6])]
    for n in span_lens:
        parts.append(torch.full((n,), IMAGE_TOKEN))
    parts.append(torch.tensor([7, 8, 9]))
    input_ids = torch.cat(parts).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": pixel_values,
        "image_sizes": sizes,
    }, span_lens


@torch.no_grad()
def test_detects_llava_onevision(model):
    from fastvision.adapters import detect_adapter

    assert detect_adapter(model).name == "llava_onevision"


@torch.no_grad()
def test_generate_runs_and_prunes(model):
    inputs, span_lens = make_inputs(model)
    FastVisionWrapper(model, keep_ratio=0.4, min_keep=1)
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    assert out.shape[-1] >= 5
    stats = model.fastvision.stats
    assert stats["visual_tokens_in"] == sum(span_lens)
    assert stats["visual_tokens_out"] == sum(max(1, round(n * 0.4)) for n in span_lens)


@torch.no_grad()
def test_anyres_variable_spans(model):
    inputs, span_lens = make_inputs(model, image_sizes=((16, 16), (32, 32)))
    assert len(set(span_lens)) > 1, "test requires variable spans"
    FastVisionWrapper(model, keep_ratio=0.4, min_keep=1)
    model.generate(**inputs, max_new_tokens=3, do_sample=False)
    stats = model.fastvision.stats
    assert stats["visual_tokens_in"] == sum(span_lens)
    assert stats["visual_tokens_out"] == sum(max(1, round(n * 0.4)) for n in span_lens)


@torch.no_grad()
def test_forward_prunes_seq_len(model):
    inputs, span_lens = make_inputs(model)
    base_len = model(**inputs).logits.shape[1]
    FastVisionWrapper(model, keep_tokens=2, min_keep=1)
    pruned_len = model(**inputs).logits.shape[1]
    assert base_len == inputs["input_ids"].shape[1]
    assert pruned_len == base_len - (sum(span_lens) - 2 * len(span_lens))


@torch.no_grad()
def test_unwrap_restores_exactly(model):
    inputs, _ = make_inputs(model)
    base_out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    base_logits = model(**inputs).logits

    FastVisionWrapper(model, keep_ratio=0.4, min_keep=1)
    pruned_out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    model.unwrap()
    assert not hasattr(model, "fastvision")

    assert torch.equal(model.generate(**inputs, max_new_tokens=6, do_sample=False), base_out)
    assert torch.equal(model(**inputs).logits, base_logits)
    assert pruned_out.shape != base_out.shape or not torch.equal(pruned_out, base_out)
