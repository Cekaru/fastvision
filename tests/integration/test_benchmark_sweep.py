"""run_sweep end-to-end on a tiny random LLaVA with a stub processor."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import BatchFeature, LlavaConfig, LlavaForConditionalGeneration  # noqa: E402

from benchmarks.datasets import Record, Task, exact_match  # noqa: E402
from benchmarks.run import format_markdown, run_sweep  # noqa: E402

IMAGE_SIZE = 30
PATCH = 6
N_PATCHES = (IMAGE_SIZE // PATCH) ** 2
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


class StubProcessor:
    """Mimics the tiny slice of AutoProcessor the harness uses."""

    def apply_chat_template(self, messages, **kwargs):
        torch.manual_seed(2)
        input_ids = torch.cat(
            [
                torch.tensor([[1, 5, 6]]),
                torch.full((1, N_PATCHES), IMAGE_TOKEN),
                torch.tensor([[7, 8, 9]]),
            ],
            dim=1,
        )
        return BatchFeature(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            }
        )

    def batch_decode(self, ids, **kwargs):
        return [" ".join(str(t) for t in ids[0].tolist())]


def test_run_sweep_tiny(model):
    task = Task(
        "stub",
        [Record(image=None, question=f"q{i}", answers=["unlikely"]) for i in range(2)],
        exact_match,
    )
    rows = run_sweep(
        model,
        StubProcessor(),
        task,
        strategies=["divprune", "random"],
        keep_ratios=[1.0, 0.2],
        max_new_tokens=3,
        eff_iters=2,
        wrap_kwargs={"min_keep": 2},
    )
    # 1 baseline + 2 strategies x 1 pruned ratio
    assert [r["strategy"] for r in rows] == ["baseline", "divprune", "random"]
    for row in rows:
        assert row["n"] == 2
        assert row["prefill_latency_s"] > 0
    pruned = rows[1]
    assert pruned["visual_tokens_out"] == 5  # round(25 * 0.2)
    assert pruned["seq_len"] < rows[0]["seq_len"]
    # model fully restored after the sweep
    assert not hasattr(model, "fastvision")
    assert "divprune" in format_markdown(rows)
