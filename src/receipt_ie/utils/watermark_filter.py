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
def detect_visual_watermark(image_path: str) -> Tuple[bool, Optional[str]]:
    """
    Detect watermark-like overlays using pixel statistics (OpenCV).
    - High edge density + high contrast = overlay or watermark
    - Extreme smoothness = blurred watermark region
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, "Unreadable image file."

        # Resize large images for speed
        h, w = img.shape
        if h * w > 2_000_000:
            img = cv2.resize(img, (w // 2, h // 2))

        # Step 1 — Edge detection
        edges = cv2.Canny(img, 30, 100)
        edge_density = np.sum(edges > 0) / img.size

        # Step 2 — Local brightness variation
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        std_dev = np.std(blur)

        # Step 3 — Low contrast detection (common in blurred watermarks)
        contrast = img.max() - img.min()

        # Heuristic decision
        if edge_density > EDGE_DENSITY_THRESHOLD and std_dev > BRIGHTNESS_VARIATION_THRESHOLD:
            return True, f"Unnatural high-frequency texture detected (edge density={edge_density:.2f})"
        if contrast < LOW_CONTRAST_THRESHOLD and std_dev < 15:
            return True, "Low contrast / blurred text region detected (possible watermark)"
        return False, "No visual watermark indicators"
    except Exception as e:
        return False, f"Visual detection failed: {e}"


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
