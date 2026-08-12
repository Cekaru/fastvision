"""End-to-end test on a tiny randomly initialized Qwen2.5-VL (no downloads).

Same dynamic-resolution + m-rope contract as Qwen2-VL; this guards that the
thin Qwen2.5-VL adapter (different model_type, extra video kwargs) detects
and prunes correctly.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402

from fastvision import FastVisionWrapper  # noqa: E402

VOCAB = 128
IMAGE_TOKEN = 125
VIDEO_TOKEN = 126
VSTART, VEND = 123, 124
MERGE = 2
PATCH = 4
TEMPORAL = 2
PATCH_DIM = 3 * TEMPORAL * PATCH * PATCH  # flattened patch input dim


@pytest.fixture()
def model():
    cfg = Qwen2_5_VLConfig(
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "out_hidden_size": 32,  # must equal text hidden size
            "num_heads": 4,
            "in_chans": 3,
            "patch_size": PATCH,
            "temporal_patch_size": TEMPORAL,
            "spatial_merge_size": MERGE,
            "intermediate_size": 64,
            "window_size": 112,
            "fullatt_block_indexes": [1],
        },
        text_config={
            "vocab_size": VOCAB,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 512,
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        image_token_id=IMAGE_TOKEN,
        video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VSTART,
        vision_end_token_id=VEND,
    )
    torch.manual_seed(0)
    m = Qwen2_5_VLForConditionalGeneration(cfg)
    m.eval()
    return m


def make_inputs(grids=((1, 4, 4),)):
    torch.manual_seed(1)
    grid = torch.tensor(list(grids), dtype=torch.long)
    n_patches = int(grid.prod(-1).sum())
    tokens_per_image = [int(g.prod()) // MERGE**2 for g in grid]

    parts = [torch.tensor([1, 5, 6])]
    for n in tokens_per_image:
        parts += [torch.tensor([VSTART]), torch.full((n,), IMAGE_TOKEN), torch.tensor([VEND])]
    parts.append(torch.tensor([7, 8, 9]))
    input_ids = torch.cat(parts).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.randn(n_patches, PATCH_DIM),
        "image_grid_thw": grid,
        "mm_token_type_ids": (input_ids == IMAGE_TOKEN).to(torch.int),
    }


@torch.no_grad()
def test_detects_qwen2_5_vl(model):
    from fastvision.adapters import detect_adapter

    assert detect_adapter(model).name == "qwen2_5_vl"


@torch.no_grad()
def test_generate_runs_and_prunes(model):
    FastVisionWrapper(model, keep_ratio=0.5, min_keep=1)
    inputs = make_inputs(grids=((1, 4, 4),))  # 4 visual tokens
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    assert out.shape[-1] >= 5
    stats = model.fastvision.stats
    assert stats["visual_tokens_in"] == 4
    assert stats["visual_tokens_out"] == 2
    assert stats["seq_len_out"] == stats["seq_len_in"] - 2


@torch.no_grad()
def test_dynamic_resolution_multiple_images(model):
    FastVisionWrapper(model, keep_ratio=0.5, min_keep=1)
    inputs = make_inputs(grids=((1, 4, 4), (1, 8, 8)))  # 4 and 16 tokens
    model.generate(**inputs, max_new_tokens=4, do_sample=False)
    stats = model.fastvision.stats
    assert stats["visual_tokens_in"] == 20
    assert stats["visual_tokens_out"] == 10  # per-image budget: 2 + 8


@torch.no_grad()
def test_forward_prunes_seq_len(model):
    inputs = make_inputs()
    base_len = model(**inputs).logits.shape[1]
    FastVisionWrapper(model, keep_tokens=1, min_keep=1)
    pruned_len = model(**inputs).logits.shape[1]
    assert base_len == inputs["input_ids"].shape[1]
    assert pruned_len == base_len - 3


@torch.no_grad()
def test_unwrap_restores_exactly(model):
    inputs = make_inputs()
    base_out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    base_logits = model(**inputs).logits

    FastVisionWrapper(model, keep_ratio=0.5, min_keep=1)
    pruned_out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    model.unwrap()
    assert not hasattr(model, "fastvision")

    assert torch.equal(model.generate(**inputs, max_new_tokens=6, do_sample=False), base_out)
    assert torch.equal(model(**inputs).logits, base_logits)
    assert pruned_out.shape != base_out.shape or not torch.equal(pruned_out, base_out)
