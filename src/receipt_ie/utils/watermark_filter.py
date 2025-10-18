"""
watermark_filter.py
--------------------
Detects textual and visual watermarks using a trained ViT classifier
and OCR text-based keyword checks.
"""

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import re

# === Hugging Face model path ===
WM_MODEL_ID = "muhammadummerrr/vit-water-mark-detector-v1"

# === Load model once globally ===
device = "cuda" if torch.cuda.is_available() else "cpu"
wm_processor = ViTImageProcessor.from_pretrained(WM_MODEL_ID)
wm_model = ViTForImageClassification.from_pretrained(WM_MODEL_ID).to(device).eval()

# === Textual watermark keywords ===
TEXTUAL_WATERMARKS = [
    "confidential", "sample", "demo", "void", "training",
    "watermark", "practice", "test", "unofficial", "do not copy",
]

def check_textual_watermark(ocr_text: str) -> (bool, str):
    text = ocr_text.lower()
    for w in TEXTUAL_WATERMARKS:
        if w in text:
            return True, f"textual watermark detected ('{w}')"
    return False, ""


def check_visual_watermark(image_path: str, threshold: float = 0.65) -> (bool, str):
    """Use ViT classifier to predict visual watermark probability."""
    im = Image.open(image_path).convert("RGB")
    inputs = wm_processor(images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = wm_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    clean_prob, wm_prob = probs[0], probs[1]
    if wm_prob >= threshold:
        return True, f"visual watermark detected (prob={wm_prob:.2f})"
    return False, f"no watermark detected (prob={wm_prob:.2f})"


def check_watermark(image_path: str, ocr_text: str = "") -> (bool, str):
    """
    Combined filter:
    1️⃣ Check textual watermarks in OCR text.
    2️⃣ Check visual watermark probability via ViT classifier.
    """
    # textual pass
    if ocr_text:
        found, reason = check_textual_watermark(ocr_text)
        if found:
            return True, f"Textual watermark → {reason}"

    # visual pass
    found, reason = check_visual_watermark(image_path)
    if found:
        return True, f"Visual watermark → {reason}"

    return False, "No watermark detected"
