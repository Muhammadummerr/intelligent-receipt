# inference_layoutlmv3_fullstack.py
# ---------------------------------
# Use with your project structure:
#   src/
#     receipt_ie/
#       data/
#         dataset_infer.py
#       utils/
#         decode.py
#         postproc.py
#
# Example run:
#   python inference_layoutlmv3_fullstack.py

import os, sys, json, torch
from typing import Dict
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

# ---- Project imports (keep these paths correct) ----
sys.path.append(os.path.abspath("."))  # so "src" is importable
from src.receipt_ie.data.dataset_infer import ReceiptInferenceDataset
from src.receipt_ie.utils.decode import group_bio
from src.receipt_ie.utils.postproc import (
    clean_company, extract_best_date, pick_total_from_lines, norm_total
)

# ===========================
# CONFIG
# ===========================
# You can point MODEL_ID to a HF repo (e.g., "muhammadummerrr/layoutlmv3-receipt-epochs-20")
# or to a local folder (e.g., "/kaggle/temp/outputs_layoutlmv3/final_model")
MODEL_ID   = "muhammadummerrr/layoutlmv3-receipt-epochs-20"
IMG_DIR    = "/kaggle/input/receipt-dataset/test/img"
BOX_DIR    = "/kaggle/input/receipt-dataset/test/box"
MAX_LEN    = 512
LIMIT      = None        # e.g., 20 to limit processed samples, or None for all
OUT_PATH   = "/kaggle/working/preds_layoutlmv3.jsonl"

# ===========================
# LOAD MODEL/PROCESSOR
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}")

processor = LayoutLMv3Processor.from_pretrained(MODEL_ID, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_ID).to(device).eval()

# id2label may be string-keyed in config; coerce to int keys
id2label = getattr(model.config, "id2label", None)
if not id2label:
    # fallback (shouldn't happen if trained with your LABELS)
    id2label = {
        0: "O",
        1: "B-COMPANY", 2: "I-COMPANY",
        3: "B-DATE",    4: "I-DATE",
        5: "B-ADDRESS", 6: "I-ADDRESS",
        7: "B-TOTAL",   8: "I-TOTAL",
    }
else:
    id2label = {int(k): v for k, v in id2label.items()}

# ===========================
# DATASET
# ===========================
inf_ds = ReceiptInferenceDataset(
    img_dir=IMG_DIR,
    box_dir=BOX_DIR,
    processor=processor,
    max_seq_len=MAX_LEN,
)

print(f"📦 Found {len(inf_ds)} receipts.")

# ===========================
# INFERENCE HELPERS
# ===========================
@torch.no_grad()
def infer_item(item) -> Dict[str, str]:
    """
    item fields:
      - input_ids, attention_mask, bbox, pixel_values, word_ids, orig_words, id, line_texts
    """
    # Add batch dimension + move to device
    inputs = {
        "input_ids": item["input_ids"].unsqueeze(0).to(device),
        "bbox": item["bbox"].unsqueeze(0).to(device),
        "pixel_values": item["pixel_values"].unsqueeze(0).to(device),
        "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
    }

    outputs = model(**inputs)
    pred_ids = outputs.logits.argmax(-1).squeeze(0).cpu().tolist()

    # BIO → fields (raw)
    raw = group_bio(item["orig_words"], pred_ids, id2label=id2label)

    # ---- Post-processing & fallbacks ----
    # COMPANY
    if raw.get("company"):
        raw["company"] = clean_company(raw["company"])

    # DATE: if empty or messy, recover from all lines
    if not raw.get("date"):
        raw["date"] = extract_best_date(" ".join(item["line_texts"]))

    # TOTAL: prefer model output; if empty, pick from lines; always normalize
    if not raw.get("total"):
        # returns a string like "85.20" or "1,234.00"; might be empty
        total_candidate = pick_total_from_lines(item["line_texts"])
        raw["total"] = norm_total(total_candidate) if total_candidate else ""
    else:
        raw["total"] = norm_total(raw["total"])

    # ADDRESS: light tidy (optional – you can add a stronger normalizer if you like)
    if raw.get("address"):
        raw["address"] = " ".join(raw["address"].split())

    return {
        "company": raw.get("company", "").strip(),
        "date": raw.get("date", "").strip(),
        "address": raw.get("address", "").strip(),
        "total": raw.get("total", "").strip(),
    }

# ===========================
# RUN
# ===========================
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
n = len(inf_ds) if LIMIT is None else min(LIMIT, len(inf_ds))
w = open(OUT_PATH, "w", encoding="utf-8")

print(f"🚀 Running inference on {n} receipts...")
hit = 0
for i in range(n):
    item = inf_ds[i]
    try:
        pred = infer_item(item)
        w.write(json.dumps({"id": item["id"], "prediction": pred}, ensure_ascii=False) + "\n")
        # Pretty print a few:
        if hit < 5:
            print(f"\n🧾 {item['id']}\n{json.dumps(pred, indent=2, ensure_ascii=False)}")
            hit += 1
    except Exception as e:
        print(f"⚠️ Error on {item['id']}: {e}")

w.close()
print(f"✅ Saved predictions to {OUT_PATH}")
