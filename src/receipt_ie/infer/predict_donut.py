import os
import json
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from peft import PeftModel

@torch.inference_mode()
def run_inference_single(image_path: str, model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔍 Running Donut inference for {image_path}")
    processor = DonutProcessor.from_pretrained(model_dir)

    # load base model + LoRA adapter
    base_model = VisionEncoderDecoderModel.from_pretrained("Bennet1996/donut-small")
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    prompt = "<s_receipt>"
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=256,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )

    seq = processor.batch_decode(outputs, skip_special_tokens=True)[0].replace(prompt, "").strip()
    try:
        extracted = json.loads(seq)
    except Exception:
        extracted = {"company": "", "date": "", "address": "", "total": ""}

    for k in ["company", "date", "address", "total"]:
        extracted.setdefault(k, "")

    return extracted, seq
