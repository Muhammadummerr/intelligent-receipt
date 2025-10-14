# import os, json, re, argparse
# from tqdm import tqdm
# import torch
# from transformers import LayoutLMv3Processor, AutoModelForTokenClassification
# from ..data.dataset_infer import ReceiptInferenceDataset
# from ..utils.decode import group_bio
# from ..utils.postproc import (
#     norm_spaces,
#     extract_best_date,
#     soft_total_norm,
#     pick_total_from_lines,
#     clean_company,
# )

# @torch.no_grad()
# def main(args):
#     os.makedirs(args.out_dir, exist_ok=True)

#     # --- Load model & processor ---
#     model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
#     try:
#         processor = LayoutLMv3Processor.from_pretrained(args.model_dir, apply_ocr=False)
#     except Exception:
#         proc_dir = args.processor_dir or "microsoft/layoutlmv3-base"
#         processor = LayoutLMv3Processor.from_pretrained(proc_dir, apply_ocr=False)

#     model.eval().to(args.device)

#     # --- Dataset ---
#     ds = ReceiptInferenceDataset(
#         img_dir=os.path.join(args.data_root, "test", "img"),
#         box_dir=os.path.join(args.data_root, "test", "box"),
#         processor=processor,
#         max_seq_len=args.max_seq_len,
#     )

#     all_out = []
#     for i in tqdm(range(len(ds))):
#         sample = ds[i]
#         sid = sample["id"]
#         word_ids = sample["word_ids"].numpy().tolist()
#         orig_words = sample["orig_words"]

#         model_inputs = {
#             k: v.unsqueeze(0).to(args.device)
#             for k, v in sample.items()
#             if k in ("input_ids", "attention_mask", "bbox", "pixel_values")
#         }

#         logits = model(**model_inputs).logits
#         pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()

#         # map token-level → word-level
#         word_to_labels = {}
#         for token_idx, w_id in enumerate(word_ids):
#             if w_id == -1:
#                 continue
#             word_to_labels.setdefault(w_id, []).append(pred_ids[token_idx])

#         import collections
#         word_level_ids = []
#         for w_id in range(len(orig_words)):
#             labs = word_to_labels.get(w_id, [])
#             word_level_ids.append(
#                 collections.Counter(labs).most_common(1)[0][0] if labs else 0
#             )

#         # --- BIO grouping ---
#         fields = group_bio(orig_words, word_level_ids)

#         # --- Post-processing ---
#         raw_company = fields.get("company", "")
#         raw_date = fields.get("date", "")
#         raw_address = fields.get("address", "")
#         raw_total = fields.get("total", "")

#         company = clean_company(raw_company)

#         # DATE: extract a single date token from span; fallback to any date in the whole text
#         date = extract_best_date(raw_date)
#         if not date:
#             date = extract_best_date(" ".join(orig_words))
#         if not date:
#             # 🌟 NEW: final fallback – try scanning full receipt text lines
#             date = extract_best_date(" ".join(sample.get("line_texts", [])))

#         # TOTAL: prefer number inside span; otherwise pick from all text with keyword heuristics
#         total = soft_total_norm(raw_total)
#         if not total:
#             total = pick_total_from_lines(sample.get("line_texts", []))

#         # ADDRESS: normalize spaces
#         address = norm_spaces(raw_address)


#         # --- Fallbacks ---
#         if not company:
#             company = clean_company(" ".join(orig_words[:50]))
#         if not address:
#             # 🌟 Try detecting address-like pattern in the full OCR text, not just tokens
#             joined = " ".join(sample.get("line_texts", []))
#             m = re.search(
#                 r"\b(?:NO\.?\s*\d+|LOT\s*\d+|JALAN|TAMAN|SEKSYEN|BANDAR|MALAYSIA|SELANGOR|KUALA\s*LUMPUR)\b.*",
#                 joined,
#                 flags=re.I
#             )
#             if m:
#                 address = norm_spaces(m.group(0))

#         fields_norm = {
#             "company": norm_spaces(company),
#             "date": norm_spaces(date),
#             "address": norm_spaces(address),
#             "total": norm_spaces(total),
#         }

#         out_path = os.path.join(args.out_dir, f"{sid}.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(fields_norm, f, ensure_ascii=False, indent=2)

#         all_out.append({"id": sid, **fields_norm})

#     with open(os.path.join(args.out_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
#         for r in all_out:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

# # ----------------------------------------------------------------------
# # 🔹 Single-image inference helper for run_pipeline.py
# # ----------------------------------------------------------------------
# def run_inference_single(image_path: str, model_dir: str):
#     """
#     Run LayoutLMv3 on a single image and return:
#       - extracted_fields (dict)
#       - ocr_text (string)
#     """
#     from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
#     from PIL import Image
#     import torch
#     import json
#     import os

#     # --- Load model + processor ---
#     processor = LayoutLMv3Processor.from_pretrained(model_dir)
#     model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
#     model.eval()

#     # --- Load image ---
#     image = Image.open(image_path).convert("RGB")

#     # --- Find corresponding OCR box file ---
#     stem = os.path.splitext(os.path.basename(image_path))[0]
#     box_path = f"/kaggle/input/receipt-dataset/test/box/{stem}.txt"
#     if not os.path.exists(box_path):
#         raise FileNotFoundError(f"Missing OCR boxes for {stem}")

#     with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
#         ocr_lines = f.readlines()

#     words, boxes = [], []
#     for line in ocr_lines:
#         parts = line.strip().split(",")
#         if len(parts) < 9:
#             continue
#         coords = list(map(int, parts[:8]))
#         text = ",".join(parts[8:]).strip()
#         words.append(text)
#         # approximate bounding box normalization (if LayoutLMv3 expects 0–1000)
#         xmin, ymin, xmax, ymax = coords[0], coords[1], coords[4], coords[5]
#         boxes.append([xmin, ymin, xmax, ymax])

#     # --- Prepare inputs for LayoutLMv3 ---
#     inputs = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predictions = outputs.logits.argmax(-1).squeeze().tolist()
#         label_map = model.config.id2label

#     # --- Build extracted fields ---
#     tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
#     fields = {"company": "", "date": "", "address": "", "total": ""}

#     current_field = None
#     for token, pred in zip(tokens, predictions):
#         label = label_map[pred]
#         if label.startswith("B-"):
#             current_field = label.split("-")[1].lower()
#             fields[current_field] += " " + token.replace("##", "")
#         elif label.startswith("I-") and current_field:
#             fields[current_field] += " " + token.replace("##", "")
#         else:
#             current_field = None

#     # --- Join and clean up ---
#     for k, v in fields.items():
#         fields[k] = v.strip().replace(" ##", "")

#     # --- Return both fields and raw OCR text ---
#     ocr_text = "\n".join(words)
#     return fields, ocr_text


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_root", required=True)
#     ap.add_argument("--model_dir", required=True)
#     ap.add_argument("--out_dir", default="./preds")
#     ap.add_argument("--batch_size", type=int, default=2)
#     ap.add_argument("--max_seq_len", type=int, default=512)
#     ap.add_argument("--device", default="cuda")
#     ap.add_argument("--processor_dir", default=None)
#     args = ap.parse_args()
#     main(args)


# src/receipt_ie/infer/predict_layoutlmv3.py
import os, json, re, argparse, torch, collections
from tqdm import tqdm
from transformers import LayoutLMv3Processor, AutoModelForTokenClassification, LayoutLMv3ForTokenClassification
from ..data.dataset_infer import ReceiptInferenceDataset
from ..utils.decode import group_bio
from ..utils.postproc import (
    norm_spaces,
    extract_best_date,
    soft_total_norm,
    pick_total_from_lines,
    clean_company,
)
from PIL import Image

@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load model & processor ---
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(args.device).eval()
    processor = LayoutLMv3Processor.from_pretrained(args.model_dir, apply_ocr=False)

    # --- Dataset ---
    ds = ReceiptInferenceDataset(
        img_dir=os.path.join(args.data_root, "test", "img"),
        box_dir=os.path.join(args.data_root, "test", "box"),
        processor=processor,
        max_seq_len=args.max_seq_len,
    )

    all_out = []
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        sid = sample["id"]
        word_ids = sample.get("word_ids", []).numpy().tolist()
        orig_words = sample["orig_words"]

        inputs = {
            k: v.unsqueeze(0).to(args.device)
            for k, v in sample.items()
            if k in ("input_ids", "attention_mask", "bbox", "pixel_values")
        }

        logits = model(**inputs).logits
        pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()

        # --- Map token → word level ---
        word_to_labels = {}
        for token_idx, w_id in enumerate(word_ids):
            if w_id == -1:
                continue
            word_to_labels.setdefault(w_id, []).append(pred_ids[token_idx])

        word_level_ids = [
            collections.Counter(word_to_labels.get(w, [0])).most_common(1)[0][0]
            for w in range(len(orig_words))
        ]

        # --- BIO grouping ---
        fields = group_bio(orig_words, word_level_ids)

        # --- Post-processing ---
        raw_company = fields.get("company", "")
        raw_date = fields.get("date", "")
        raw_address = fields.get("address", "")
        raw_total = fields.get("total", "")

        company = clean_company(raw_company) or clean_company(" ".join(orig_words[:50]))
        date = extract_best_date(raw_date) or extract_best_date(" ".join(orig_words))
        total = soft_total_norm(raw_total) or pick_total_from_lines(sample.get("line_texts", []))
        address = norm_spaces(raw_address)
        if not address:
            joined = " ".join(sample.get("line_texts", []))
            m = re.search(r"\b(NO\.?\s*\d+|LOT\s*\d+|JALAN|TAMAN|BANDAR|SELANGOR|MALAYSIA|KUALA\s*LUMPUR)\b.*", joined, re.I)
            if m:
                address = norm_spaces(m.group(0))

        fields_norm = {
            "company": norm_spaces(company),
            "date": norm_spaces(date),
            "address": norm_spaces(address),
            "total": norm_spaces(total),
        }

        out_path = os.path.join(args.out_dir, f"{sid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fields_norm, f, ensure_ascii=False, indent=2)
        all_out.append({"id": sid, **fields_norm})

    with open(os.path.join(args.out_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
        for r in all_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ #
# Single-image inference helper for run_pipeline.py
# ------------------------------------------------------------------ #
@torch.no_grad()
def run_inference_single(image_path: str, model_dir: str, device="cuda"):
    processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir).to(device).eval()

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    stem = os.path.splitext(os.path.basename(image_path))[0]
    box_path = os.path.join(os.path.dirname(image_path), "../box", f"{stem}.txt")
    box_path = os.path.abspath(box_path)

    with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
        ocr_lines = f.readlines()

    words, boxes = [], []
    for line in ocr_lines:
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue
        coords = list(map(int, parts[:8]))
        text = ",".join(parts[8:]).strip()
        words.append(text)
        xmin, ymin, xmax, ymax = coords[0], coords[1], coords[4], coords[5]
        boxes.append([
            int(xmin / W * 1000),
            int(ymin / H * 1000),
            int(xmax / W * 1000),
            int(ymax / H * 1000),
        ])

    inputs = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True).to(device)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    label_map = model.config.id2label

    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    fields = {"company": "", "date": "", "address": "", "total": ""}
    current_field = None
    for token, pred in zip(tokens, predictions):
        label = label_map[pred]
        if label.startswith("B-"):
            current_field = label.split("-")[1].lower()
            fields[current_field] += " " + token.replace("##", "")
        elif label.startswith("I-") and current_field:
            fields[current_field] += " " + token.replace("##", "")
        else:
            current_field = None

    # Normalize
    for k, v in fields.items():
        fields[k] = norm_spaces(v)
    fields["company"] = clean_company(fields["company"])
    fields["date"] = extract_best_date(fields["date"]) or ""
    fields["total"] = soft_total_norm(fields["total"]) or ""

    ocr_text = "\n".join(words)
    return fields, ocr_text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", default="./preds")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args)
