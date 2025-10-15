import os, json, re, argparse, torch, collections
from tqdm import tqdm
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import easyocr

from ..data.dataset_infer import ReceiptInferenceDataset
from ..utils.decode import group_bio
from ..utils.postproc import (
    norm_spaces,
    extract_best_date,
    soft_total_norm,
    pick_total_from_lines,
    clean_company,
)

# ---------------------------------------------------------------------- #
#  LOAD MODEL + PROCESSOR
# ---------------------------------------------------------------------- #
def load_model_and_processor(model_dir_or_hub: str, device="cuda"):
    """
    Load LayoutLMv3 model + processor from local folder or Hugging Face Hub.
    Keeps apply_ocr=False (so model matches training),
    but attaches an EasyOCR reader for fallback OCR extraction.
    """
    if os.path.isdir(model_dir_or_hub):
        print(f"📦 Loading local model from {model_dir_or_hub}")
    else:
        print(f"☁️ Loading model from Hugging Face Hub: {model_dir_or_hub}")

    processor = LayoutLMv3Processor.from_pretrained(model_dir_or_hub, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir_or_hub).to(device)
    model.eval()

    # 🔹 Attach EasyOCR reader for inference OCR
    processor.easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return processor, model


# ---------------------------------------------------------------------- #
#  BATCH INFERENCE
# ---------------------------------------------------------------------- #
@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load model & processor (local or HF) ---
    processor, model = load_model_and_processor(args.model_dir, args.device)

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
        word_ids = sample["word_ids"].numpy().tolist()
        orig_words = sample["orig_words"]

        inputs = {
            k: v.unsqueeze(0).to(args.device)
            for k, v in sample.items()
            if k in ("input_ids", "attention_mask", "bbox", "pixel_values")
        }

        logits = model(**inputs).logits
        pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()

        # --- Token → Word aggregation ---
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
            m = re.search(
                r"\b(NO\.?\s*\d+|LOT\s*\d+|JALAN|TAMAN|BANDAR|SELANGOR|MALAYSIA|KUALA\s*LUMPUR)\b.*",
                joined,
                re.I,
            )
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


# ---------------------------------------------------------------------- #
#  SINGLE IMAGE INFERENCE (with dynamic OCR fallback)
# ---------------------------------------------------------------------- #
@torch.no_grad()
def run_inference_single(image_path: str, model_dir: str, device="cuda"):
    """
    Run LayoutLMv3 inference on a single receipt image.
    If OCR box file is missing, extract text and boxes automatically using EasyOCR.
    """
    print(f"🔍 Running inference for {os.path.basename(image_path)}")

    # --- Load model + processor ---
    processor, model = load_model_and_processor(model_dir, device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    stem = os.path.splitext(os.path.basename(image_path))[0]
    box_path = os.path.abspath(os.path.join(os.path.dirname(image_path), "../box", f"{stem}.txt"))

    words, boxes = [], []

    # --- Load precomputed OCR boxes if available ---
    if os.path.exists(box_path):
        print(f"📄 Using OCR boxes from: {box_path}")
        with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                coords = list(map(int, parts[:8]))
                text = ",".join(parts[8:]).strip()
                if not text:
                    continue
                xmin, ymin, xmax, ymax = coords[0], coords[1], coords[4], coords[5]
                boxes.append([
                    int(xmin / W * 1000),
                    int(ymin / H * 1000),
                    int(xmax / W * 1000),
                    int(ymax / H * 1000),
                ])
                words.append(text)
    else:
        # --- Fallback: run EasyOCR if no box file is found ---
        print("🧠 No precomputed OCR boxes found — running EasyOCR...")
        reader = getattr(processor, "easyocr_reader", None)
        if reader is None:
            reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

        results = reader.readtext(image_path, detail=1, paragraph=False)
        if not results:
            raise RuntimeError("❌ OCR failed: no text detected in image.")

        for (bbox, text, conf) in results:
            if not text.strip():
                continue
            ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = bbox
            xmin, ymin = min(x1, x2, x3, x4), min(y1, y2, y3, y4)
            xmax, ymax = max(x1, x2, x3, x4), max(y1, y2, y3, y4)
            boxes.append([
                int(xmin / W * 1000),
                int(ymin / H * 1000),
                int(xmax / W * 1000),
                int(ymax / H * 1000),
            ])
            words.append(text.strip())

        # --- Optional: cache OCR results ---
        os.makedirs("./ocr_cache", exist_ok=True)
        cache_path = f"./ocr_cache/{stem}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"words": words, "boxes": boxes}, f, ensure_ascii=False, indent=2)

    if not words:
        raise RuntimeError("❌ No text found for LayoutLMv3 input.")

    # --- Run LayoutLMv3 model ---
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

    # --- Normalize & clean ---
    for k, v in fields.items():
        fields[k] = norm_spaces(v)
    fields["company"] = clean_company(fields["company"])
    fields["date"] = extract_best_date(fields["date"]) or ""
    fields["total"] = soft_total_norm(fields["total"]) or ""

    ocr_text = "\n".join(words)
    print("✅ Inference complete.")
    return fields, ocr_text


# ---------------------------------------------------------------------- #
#  ENTRY POINT
# ---------------------------------------------------------------------- #
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
