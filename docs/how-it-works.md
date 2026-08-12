# How it works

## The problem

Multimodal LLMs encode an image into hundreds to thousands of visual tokens:
LLaVA-1.5 at 336px produces 576, LLaVA-1.6 anyres and Qwen2-VL dynamic
resolution produce 2k–5k+. Those tokens dominate the prefill sequence and the
KV cache — yet most encode redundant background and repeated texture.
Removing the redundant ones before the LLM barely touches accuracy but
linearly cuts attention cost and cache size.

## The algorithm: DivPrune

FastVision's default strategy solves a **Max-Min Diversity Problem**: select
the subset of visual tokens that maximizes the *minimum* pairwise distance
inside the subset — i.e. maximally diverse, coverage-preserving tokens.

It is solved greedily with farthest-point sampling, fully vectorized on GPU:

1. L2-normalize features; distance = 1 − cosine similarity.
2. Seed with the highest-norm token.
3. Repeat K times: pick the token whose minimum distance to the selected set
   is largest, then update the running `min_dist` in O(N).

That is O(N·K) per image — milliseconds even at N = 5000.

Why it is the right flagship for a *wrapper*:

- **Training-free** — pure inference-time operation.
- **Attention-free** — operates on token *features*, not attention weights,
  so it composes with FlashAttention / SDPA (attention-based pruning breaks
  when attention matrices are never materialized).
- **Clean interception point** — the projected visual embeddings are a single
  well-defined tensor before they enter the LLM.

## The interception

Pruning image features alone is not enough: the model scatters them into
`inputs_embeds` at `<image>` placeholder positions. Dropping K of N features
requires dropping the matching placeholders and rebuilding `attention_mask`
and `position_ids` to the new length. FastVision:

1. Wraps `generate`/`forward`; on a multimodal prefill it computes the
   projected image features and runs the pruner → `keep_index`.
2. Drops the un-kept placeholder positions and scatters the kept (or merged)
   features into the surviving slots.
3. Rebuilds the attention mask — and for Qwen2-VL, gathers the precomputed 3D
   m-rope `position_ids` at kept positions so surviving tokens retain their
   original spatial coordinates.

The KV cache then follows naturally through generation at the shorter
length. `unwrap()` removes the hooks and restores byte-identical behavior.
