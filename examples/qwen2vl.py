"""Qwen2-VL + FastVision in 3 lines.

Dynamic-resolution images and m-rope bookkeeping are handled by the adapter:
kept visual tokens retain their original 3D rope coordinates.
"""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from fastvision import FastVisionWrapper

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = FastVisionWrapper(model, keep_ratio=0.1)  # <- the only change

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
print(f"tokens: {model.fastvision.stats}")
