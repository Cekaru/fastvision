"""FastVision quickstart: prune visual tokens on LLaVA-1.5 in 3 lines.

Requires a GPU with ~15 GB free (or pass device_map/quantization yourself).
"""

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from fastvision import FastVisionWrapper

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto"
)

# --- the 3 lines ---------------------------------------------------------
model = FastVisionWrapper(model, keep_ratio=0.1)  # 576 -> ~58 visual tokens
# -------------------------------------------------------------------------

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "USER: <image>\nWhat is shown in this image? ASSISTANT:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.decode(out[0], skip_special_tokens=True))
print("token savings:", model.fastvision.stats)

model.unwrap()  # restores the original model exactly
