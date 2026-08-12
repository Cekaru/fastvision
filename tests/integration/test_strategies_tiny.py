"""Every registered strategy runs end-to-end on a tiny LLaVA."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlavaConfig, LlavaForConditionalGeneration  # noqa: E402

from fastvision import FastVisionWrapper  # noqa: E402
from fastvision.pruners import PRUNERS  # noqa: E402

IMAGE_SIZE = 30
PATCH = 6
N_PATCHES = (IMAGE_SIZE // PATCH) ** 2  # 25
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


def make_inputs():
    torch.manual_seed(1)
    input_ids = torch.cat(
        [
            torch.tensor([[1, 5, 6]]),
            torch.full((1, N_PATCHES), IMAGE_TOKEN),
            torch.tensor([[7, 8, 9, 10]]),
        ],
        dim=1,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
    }


@pytest.mark.parametrize("strategy", sorted(PRUNERS))
@torch.no_grad()
def test_strategy_generates_and_prunes(model, strategy):
    FastVisionWrapper(model, keep_tokens=6, min_keep=2, strategy=strategy)
    out = model.generate(**make_inputs(), max_new_tokens=4, do_sample=False)
    assert out.shape[-1] >= 4
    stats = model.fastvision.stats
    assert stats["visual_tokens_in"] == N_PATCHES
    assert stats["visual_tokens_out"] == 6
    assert stats["seq_len_out"] == stats["seq_len_in"] - (N_PATCHES - 6)


@torch.no_grad()
def test_strategies_differ_from_each_other(model):
    # sanity: merger output differs from pure selection at the same budget
    inputs = make_inputs()
    logits = {}
    for strategy in ("divprune", "tome"):
        FastVisionWrapper(model, keep_tokens=6, min_keep=2, strategy=strategy)
        logits[strategy] = model(**inputs).logits
        model.unwrap()
    assert not torch.allclose(logits["divprune"], logits["tome"])
