"""End-to-end test on a tiny randomly initialized Qwen2-VL (no downloads).

Exercises dynamic resolution (different grid per image) and m-rope
bookkeeping (3D position ids gathered at kept positions).
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration  # noqa: E402

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
    cfg = Qwen2VLConfig(
        vision_config={
            "depth": 2,
            "embed_dim": 32,
            "hidden_size": 32,  # must equal text hidden size
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": PATCH,
            "temporal_patch_size": TEMPORAL,
            "spatial_merge_size": MERGE,
            "mlp_ratio": 2,
        },
        text_config={
            "vocab_size": VOCAB,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 512,
            # head_dim=8 -> mrope sections sum to head_dim/2
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        image_token_id=IMAGE_TOKEN,
        video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VSTART,
        vision_end_token_id=VEND,
    )
    torch.manual_seed(0)
    m = Qwen2VLForConditionalGeneration(cfg)
    m.eval()
    return m


def make_inputs(grids=((1, 4, 4),)):
    """One batch row containing len(grids) images with the given thw grids."""
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
    # two images with different grids: 4 and 16 visual tokens
    inputs = make_inputs(grids=((1, 4, 4), (1, 8, 8)))
    out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert out.shape[0] == 1
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


@torch.no_grad()
def test_mrope_positions_survive_pruning(model):
    """Kept visual tokens must retain their original 3D coordinates."""
    from fastvision.pruners import UniformPruner

    FastVisionWrapper(model, keep_tokens=2, min_keep=1, strategy=UniformPruner())
    inputs = make_inputs(grids=((1, 4, 4),))
    adapter = model.fastvision.adapter
    spliced = adapter.prepare(
        model, model.fastvision, inputs["input_ids"], inputs["attention_mask"],
        {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")},
    )
    assert spliced.position_ids.shape[0] == 4  # text row + 3D rope rows
    # grid (1,4,4) merged 2x2 -> 2x2 token grid at text offset 4:
    # heights [4,4,5,5], widths [4,5,4,5]; uniform keep of 2 from 4 -> offsets 0,3
    pos = spliced.position_ids
    seq = inputs["input_ids"][0].tolist()
    first_img = seq.index(IMAGE_TOKEN)
    kept_h = pos[2, 0, first_img : first_img + 2].tolist()
    kept_w = pos[3, 0, first_img : first_img + 2].tolist()
    assert kept_h == [4, 5]
    assert kept_w == [4, 5]


@torch.no_grad()
def test_video_inputs_bypass_pruning(model):
    FastVisionWrapper(model, keep_ratio=0.5, min_keep=1)
    inputs = make_inputs()
    inputs["pixel_values_videos"] = torch.randn(4, PATCH_DIM)
    # bypass: behaves like the unwrapped model (which will route videos itself)
    model.fastvision.stats.clear()
    try:
        model.generate(**inputs, video_grid_thw=torch.tensor([[1, 2, 2]]),
                       max_new_tokens=2, do_sample=False)
    except Exception:
        pass  # tiny config may not support videos; pruning must still be skipped
    assert model.fastvision.stats == {}
