import os, json, re, argparse
from tqdm import tqdm
import torch
from transformers import LayoutLMv3Processor, AutoModelForTokenClassification
from ..data.dataset_infer import ReceiptInferenceDataset
from ..utils.decode import group_bio
from ..utils.postproc import (
    norm_spaces,
    extract_best_date,
    soft_total_norm,
    pick_total_from_lines,
    clean_company,
)

@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load model & processor ---
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    try:
        processor = LayoutLMv3Processor.from_pretrained(args.model_dir, apply_ocr=False)
    except Exception:
        proc_dir = args.processor_dir or "microsoft/layoutlmv3-base"
        processor = LayoutLMv3Processor.from_pretrained(proc_dir, apply_ocr=False)

    model.eval().to(args.device)

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

        model_inputs = {
            k: v.unsqueeze(0).to(args.device)
            for k, v in sample.items()
            if k in ("input_ids", "attention_mask", "bbox", "pixel_values")
        }

        logits = model(**model_inputs).logits
        pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()

        # map token-level → word-level
        word_to_labels = {}
        for token_idx, w_id in enumerate(word_ids):
            if w_id == -1:
                continue
            word_to_labels.setdefault(w_id, []).append(pred_ids[token_idx])

        import collections
        word_level_ids = []
        for w_id in range(len(orig_words)):
            labs = word_to_labels.get(w_id, [])
            word_level_ids.append(
                collections.Counter(labs).most_common(1)[0][0] if labs else 0
            )

        # --- BIO grouping ---
        fields = group_bio(orig_words, word_level_ids)

        # --- Post-processing ---
        raw_company = fields.get("company", "")
        raw_date = fields.get("date", "")
        raw_address = fields.get("address", "")
        raw_total = fields.get("total", "")

        company = clean_company(raw_company)
        date = extract_best_date(raw_date) or extract_best_date(" ".join(orig_words))
        total = soft_total_norm(raw_total) or pick_total_from_lines(sample.get("line_texts", []))
        address = norm_spaces(raw_address)

        # --- Fallbacks ---
        if not company:
            company = clean_company(" ".join(orig_words[:50]))
        if not address:
            joined = " ".join(orig_words)
            m = re.search(r"\d{2,4}\s*[A-Z].*(?:JALAN|TAMAN|SELANGOR|KUALA LUMPUR|MALAYSIA)", joined, flags=re.I)
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", default="./preds")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--processor_dir", default=None)
    args = ap.parse_args()
    main(args)
