"""End-to-end test on a tiny randomly initialized InternVL (no downloads)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.internvl")
from transformers import InternVLConfig, InternVLForConditionalGeneration  # noqa: E402

from fastvision import FastVisionWrapper  # noqa: E402

IMAGE_SIZE = 32
PATCH = 4
# (32/4)^2 = 64 patches, drop CLS -> 64, pixel-shuffle 0.5 -> 16 tokens/tile
N_TOKENS = 16
VOCAB = 128
IMAGE_TOKEN = VOCAB - 1


@pytest.fixture()
def model():
    cfg = InternVLConfig(
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": IMAGE_SIZE,
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
        image_token_id=IMAGE_TOKEN,
    )
    torch.manual_seed(0)
    m = InternVLForConditionalGeneration(cfg)
    m.eval()
    return m


def make_inputs(tiles: int = 1):
    torch.manual_seed(1)
    input_ids = torch.cat(
        [
            torch.tensor([[1, 5, 6]]),
            torch.full((1, N_TOKENS * tiles), IMAGE_TOKEN),
            torch.tensor([[7, 8, 9, 10]]),
        ],
        dim=1,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.randn(tiles, 3, IMAGE_SIZE, IMAGE_SIZE),
    }


@torch.no_grad()
def test_generate_runs_and_prunes(model):
    FastVisionWrapper(model, keep_ratio=0.25, min_keep=2)
    out = model.generate(**make_inputs(), max_new_tokens=4, do_sample=False)
    assert out.shape[-1] >= 4
    stats = model.fastvision.stats
    assert stats["visual_tokens_in"] == N_TOKENS
    assert stats["visual_tokens_out"] == 4
    assert stats["seq_len_out"] == stats["seq_len_in"] - (N_TOKENS - 4)


@torch.no_grad()
def test_multi_tile_budgets_per_tile(model):
    FastVisionWrapper(model, keep_tokens=4, min_keep=2)
    model.generate(**make_inputs(tiles=2), max_new_tokens=3, do_sample=False)
    stats = model.fastvision.stats
    assert stats["visual_tokens_in"] == 2 * N_TOKENS
    assert stats["visual_tokens_out"] == 2 * 4


@torch.no_grad()
def test_unwrap_restores_exactly(model):
    inputs = make_inputs()
    base_out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    base_logits = model(**inputs).logits

    FastVisionWrapper(model, keep_ratio=0.25, min_keep=2)
    model.generate(**inputs, max_new_tokens=5, do_sample=False)
    model.unwrap()

    assert torch.equal(model.generate(**inputs, max_new_tokens=5, do_sample=False), base_out)
    assert torch.equal(model(**inputs).logits, base_logits)
