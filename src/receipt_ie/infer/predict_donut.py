"""
predict_donut.py
----------------
Single-image inference for Donut (VisionEncoderDecoder) models.

Loads a fine-tuned Donut model + processor and produces:
1. Extracted JSON fields (company, date, address, total)
2. OCR text (for reasoning step)

Works with models fine-tuned using train_donut_lora_final_dp_safe.py
"""

import os
import json
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# ---------------------- Inference Function ---------------------- #
@torch.inference_mode()
def run_inference_single(image_path: str, model_dir: str):
    """
    Run Donut on a single receipt image.

    Returns:
        extracted (dict): parsed receipt fields.
        ocr_text (str): raw decoded text string (for LLM reasoning).
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ Model directory not found: {model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"📦 Loading fine-tuned Donut model from {model_dir} ...")
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()

    # Prepare image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate text
    prompt = "<s_receipt>"
    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=processor.tokenizer.model_max_length,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )

    # Decode
    seq = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    seq = seq.replace(prompt, "").strip()
    ocr_text = seq

    # Parse JSON if possible
    extracted = {}
    try:
        # Donut training target is usually JSON-like
        extracted = json.loads(seq)
    except Exception:
        # crude heuristic parsing
        for key in ["company", "date", "address", "total"]:
            m = None
            try:
                m = (
                    seq.split(f'"{key}"')[1]
                    .split(":")[1]
                    .split(",")[0]
                    .strip(' "{}')
                )
            except Exception:
                m = ""
            extracted[key] = m

    # Fill any missing keys
    for k in ["company", "date", "address", "total"]:
        extracted.setdefault(k, "")

    return extracted, ocr_text
