"""
watermark_filter.py
-------------------
Detects watermarked or tampered receipts before model inference.

Includes:
✅ Visual watermark detection (edges, contrast, overlays)
✅ Textual watermark detection (keywords like SAMPLE, VOID, etc.)
✅ Safelist of legitimate vendor names
✅ Confidence-based filtering (optional)
"""

import cv2
import numpy as np
import re
from typing import Tuple, Optional

# -----------------------------
# CONFIGURATION
# -----------------------------
# Legitimate names that must never trigger rejection
SAFE_NAMES = [
    "tan chay yee", "ahmad", "lee", "trading", "enterprise",
    "restaurant", "sdn bhd", "bhd", "enterprise", "s/b"
]

# True watermark / synthetic indicators
WATERMARK_KEYWORDS = [
    "sample", "confidential", "training",
    "void", "demo", "practice", "fake",
    "for testing", "invalid", "watermark", "do not use"
]

# Visual detection thresholds
EDGE_DENSITY_THRESHOLD = 0.12     # higher = more edges, potential overlay
BRIGHTNESS_VARIATION_THRESHOLD = 25
LOW_CONTRAST_THRESHOLD = 20


# ---------------------------------------------------
# VISUAL DETECTION
# ---------------------------------------------------
def detect_visual_watermark(image_path: str) -> tuple[bool, str]:
    import cv2, numpy as np
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, "unreadable image"

    h, w = img.shape
    img_small = cv2.resize(img, (min(w, 1024), min(h, 1024)))
    edges = cv2.Canny(img_small, 30, 100)
    blur = cv2.GaussianBlur(img_small, (5, 5), 0)

    # divide into 8×8 tiles, look for unusual texture regions
    tile_h, tile_w = img_small.shape[0]//8, img_small.shape[1]//8
    local_std = []
    for i in range(8):
        for j in range(8):
            patch = blur[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            local_std.append(np.std(patch))
    local_std = np.array(local_std)

    # big difference between median and max → probable overlay text
    if local_std.max() - np.median(local_std) > 25:
        return True, "local brightness anomaly detected (possible faint overlay)"
    return False, "no visual watermark indicators"


# ---------------------------------------------------
# TEXTUAL DETECTION
# ---------------------------------------------------
def detect_textual_watermark(ocr_text: str) -> Tuple[bool, Optional[str]]:
    """
    Reject receipt if OCR text includes known watermark indicators,
    unless a safe company name is found.
    """
    if not ocr_text.strip():
        return False, "Empty OCR text"

    text_lower = ocr_text.lower()

    # ✅ Safelist: skip watermark detection if real vendor name present
    for safe in SAFE_NAMES:
        if safe in text_lower:
            return False, f"Legitimate business term found ('{safe}')"

    # 🚫 Watermark words
    for kw in WATERMARK_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text_lower):
            return True, f"Watermark keyword detected ('{kw}')"
    return False, "No watermark keywords detected"


# ---------------------------------------------------
# CONFIDENCE-BASED DETECTION (OPTIONAL)
# ---------------------------------------------------
def detect_low_confidence(ocr_result: Optional[list]) -> Tuple[bool, Optional[str]]:
    """
    Takes EasyOCR output (list of [text, bbox, confidence]).
    If average confidence < 0.5 → possible watermark blur.
    """
    if not ocr_result:
        return False, "No OCR confidence data"
    confidences = [c for *_, c in ocr_result if isinstance(c, (float, int))]
    if not confidences:
        return False, "No numeric OCR confidences"

    avg_conf = np.mean(confidences)
    if avg_conf < 0.5:
        return True, f"Low OCR confidence ({avg_conf:.2f}) indicates possible blur/watermark"
    return False, f"Average OCR confidence {avg_conf:.2f} is acceptable"


# ---------------------------------------------------
# MAIN WRAPPER FUNCTION
# ---------------------------------------------------
def check_watermark(image_path: str,
                    ocr_text: Optional[str] = None,
                    ocr_result: Optional[list] = None) -> Tuple[bool, str]:
    """
    Unified watermark detection function.
    Returns (is_watermarked, reason)
    """
    # 1️⃣ Visual detection
    vis_flag, vis_reason = detect_visual_watermark(image_path)
    if vis_flag:
        return True, f"Visual watermark detected → {vis_reason}"

    # 2️⃣ Textual detection
    if ocr_text:
        txt_flag, txt_reason = detect_textual_watermark(ocr_text)
        if txt_flag:
            return True, f"Textual watermark detected → {txt_reason}"

    # 3️⃣ Confidence check (if OCR result provided)
    if ocr_result:
        conf_flag, conf_reason = detect_low_confidence(ocr_result)
        if conf_flag:
            return True, f"OCR quality issue → {conf_reason}"

    return False, "Clean receipt — no watermark indicators"
