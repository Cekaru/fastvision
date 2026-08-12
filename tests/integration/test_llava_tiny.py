"""End-to-end test on a tiny randomly initialized LLaVA (no downloads)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlavaConfig, LlavaForConditionalGeneration  # noqa: E402

from fastvision import FastVisionWrapper, compressed  # noqa: E402

IMAGE_SIZE = 30
PATCH = 6
N_PATCHES = (IMAGE_SIZE // PATCH) ** 2  # 25 visual tokens ("default" drops CLS)
VOCAB = 128
IMAGE_TOKEN = VOCAB - 1


@pytest.fixture()
def model():
    cfg = LlavaConfig(
        vision_config={
            "model_type": "clip_vision_model",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH,
        },
        text_config={
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": VOCAB,
            "max_position_embeddings": 256,
        },
        image_token_index=IMAGE_TOKEN,
        image_seq_length=N_PATCHES,
    )
    torch.manual_seed(0)
    m = LlavaForConditionalGeneration(cfg)
    m.eval()
    return m


def make_inputs(batch: int = 1):
    torch.manual_seed(1)
    prefix = torch.tensor([[1, 5, 6]] * batch)
    image = torch.full((batch, N_PATCHES), IMAGE_TOKEN)
    suffix = torch.tensor([[7, 8, 9, 10]] * batch)
    input_ids = torch.cat([prefix, image, suffix], dim=1)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE),
    }


@torch.no_grad()
def test_generate_runs_and_prunes(model):
    FastVisionWrapper(model, keep_ratio=0.2, min_keep=2)
    out = model.generate(**make_inputs(), max_new_tokens=5, do_sample=False)
    assert out.shape[-1] >= 5
    stats = model.fastvision.stats
    assert stats["visual_tokens_in"] == N_PATCHES
    assert stats["visual_tokens_out"] == 5  # round(25 * 0.2)
    assert stats["seq_len_out"] == stats["seq_len_in"] - (N_PATCHES - 5)


@torch.no_grad()
def test_generate_batch(model):
    FastVisionWrapper(model, keep_tokens=4, min_keep=2)
    out = model.generate(**make_inputs(batch=2), max_new_tokens=4, do_sample=False)
    assert out.shape[0] == 2
    assert model.fastvision.stats["visual_tokens_out"] == 8


@torch.no_grad()
def test_forward_prunes_seq_len(model):
    inputs = make_inputs()
    base_len = model(**inputs).logits.shape[1]
    FastVisionWrapper(model, keep_tokens=4, min_keep=2)
    pruned_len = model(**inputs).logits.shape[1]
    assert base_len == inputs["input_ids"].shape[1]
    assert pruned_len == base_len - (N_PATCHES - 4)


@torch.no_grad()
def test_unwrap_restores_exactly(model):
    inputs = make_inputs()
    base_out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    base_logits = model(**inputs).logits

    FastVisionWrapper(model, keep_ratio=0.2, min_keep=2)
    pruned_out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    model.unwrap()
    assert not hasattr(model, "fastvision")

    assert torch.equal(model.generate(**inputs, max_new_tokens=6, do_sample=False), base_out)
    assert torch.equal(model(**inputs).logits, base_logits)
    # sanity: pruning actually ran (different prompt length seen by generate)
    assert pruned_out.shape != base_out.shape or not torch.equal(pruned_out, base_out)


@torch.no_grad()
def test_enabled_toggle(model):
    inputs = make_inputs()
    base_len = model(**inputs).logits.shape[1]
    FastVisionWrapper(model, keep_tokens=4, min_keep=2, enabled=False)
    assert model(**inputs).logits.shape[1] == base_len
    model.fastvision.enabled = True
    assert model(**inputs).logits.shape[1] < base_len


@torch.no_grad()
def test_context_manager(model):
    inputs = make_inputs()
    with compressed(model, keep_ratio=0.2, min_keep=2):
        model.generate(**inputs, max_new_tokens=3, do_sample=False)
    assert not hasattr(model, "fastvision")


def test_double_wrap_rejected(model):
    FastVisionWrapper(model, keep_ratio=0.2)
    with pytest.raises(RuntimeError):
        FastVisionWrapper(model, keep_ratio=0.2)


def test_unknown_family_message():
    with pytest.raises(ValueError, match="Supported families"):
        FastVisionWrapper(torch.nn.Linear(2, 2))
