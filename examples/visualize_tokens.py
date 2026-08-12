"""Render which image patches survive pruning.

Overlays the kept-token patch grid on an image at several keep ratios and
writes a side-by-side PNG (and optionally an animated GIF of the greedy
DivPrune selection order).

Usage::

    python examples/visualize_tokens.py --image cat.jpg \
        --model llava-hf/llava-1.5-7b-hf --keep-ratios 0.5 0.2 0.1 --gif

Requires: pip install fastvision[bench] (matplotlib, pillow).
"""

from __future__ import annotations

import argparse

import torch


def kept_mask(keep_index: torch.Tensor, n: int) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    mask[keep_index] = True
    return mask


def overlay(ax, image, mask_2d, title: str):
    import numpy as np

    ax.imshow(image)
    h, w = image.size[1], image.size[0]
    gh, gw = mask_2d.shape
    dim = np.ones((h, w)) * 0.75
    ph, pw = h / gh, w / gw
    for i in range(gh):
        for j in range(gw):
            if mask_2d[i, j]:
                dim[int(i * ph) : int((i + 1) * ph), int(j * pw) : int((j + 1) * pw)] = 0.0
    ax.imshow(np.zeros((h, w)), alpha=dim, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.5, 0.2, 0.1])
    parser.add_argument("--output", default="token_selection.png")
    parser.add_argument("--gif", action="store_true", help="also write selection-order GIF")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from fastvision.pruners import DivPrune

    image = Image.open(args.image).convert("RGB")
    print(f"loading {args.model} ...")
    model = AutoModelForImageTextToText.from_pretrained(args.model).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    pixel_values = processor.image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        fn = getattr(model, "get_image_features", None) or model.model.get_image_features
        out = fn(pixel_values=pixel_values.to(args.device))
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        if isinstance(feats, (list, tuple)):
            feats = torch.stack(list(feats))
    n = feats.shape[1]
    side = int(n**0.5)
    print(f"{n} visual tokens ({side}x{side} grid)")

    pruner = DivPrune()
    cols = len(args.keep_ratios) + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    overlay(axes[0], image, torch.ones(side, side, dtype=torch.bool), f"original ({n} tokens)")
    for ax, ratio in zip(axes[1:], args.keep_ratios):
        k = max(1, round(n * ratio))
        idx = pruner.select(feats, k)[0].cpu()
        title = f"keep_ratio={ratio:g} ({k} tokens)"
        overlay(ax, image, kept_mask(idx, n).reshape(side, side), title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")

    if args.gif:
        k = max(1, round(n * min(args.keep_ratios)))
        frames = []
        # greedy selection is prefix-consistent, so select(step) shows the order
        for step in range(1, k + 1, max(1, k // 30)):
            idx = pruner.select(feats, step)[0].cpu()
            f, ax = plt.subplots(figsize=(4, 4))
            overlay(ax, image, kept_mask(idx, n).reshape(side, side), f"{step}/{k} tokens")
            f.canvas.draw()
            frames.append(Image.frombytes("RGB", f.canvas.get_width_height(),
                                          f.canvas.buffer_rgba(), "raw", "RGBA"))
            plt.close(f)
        gif_path = args.output.rsplit(".", 1)[0] + ".gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
        print(f"wrote {gif_path}")


if __name__ == "__main__":
    main()
