import os, json, argparse
from tqdm import tqdm
import torch
from transformers import LayoutLMv3Processor, AutoModelForTokenClassification

from ..data.dataset_infer import ReceiptInferenceDataset
from ..utils.decode import group_bio

# NEW: robust cleaners/selectors
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

    # choose processor (from model_dir if it contains preprocessor, else base)
    proc_dir = args.processor_dir
    if proc_dir is None:
        if os.path.isfile(os.path.join(args.model_dir, "preprocessor_config.json")):
            proc_dir = args.model_dir
        else:
            proc_dir = "microsoft/layoutlmv3-base"

    processor = LayoutLMv3Processor.from_pretrained(proc_dir, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval().to(args.device)

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
        orig_words = sample["orig_words"]  # word-level strings (before subwording)

        # build a model batch of 1
        model_inputs = {
            k: v.unsqueeze(0).to(args.device)
            for k, v in sample.items()
            if k in ("input_ids", "attention_mask", "bbox", "pixel_values")
        }

        logits = model(**model_inputs).logits  # (1, seq, n_labels)
        pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()

        # map token-level preds -> word-level preds by majority vote
        word_to_labels = {}
        for token_idx, w_id in enumerate(word_ids):
            if w_id == -1:
                continue
            word_to_labels.setdefault(w_id, []).append(pred_ids[token_idx])

        # choose most frequent label per word id
        import collections
        word_level_ids = []
        for w_id in range(len(orig_words)):
            lab_list = word_to_labels.get(w_id, [])
            if not lab_list:
                word_level_ids.append(0)  # "O"
            else:
                word_level_ids.append(collections.Counter(lab_list).most_common(1)[0][0])

        # ---------------------------
        # BIO grouping on word level
        # ---------------------------
        fields = group_bio(orig_words, word_level_ids)

        # ---------------------------
        # NEW: robust post-processing
        # ---------------------------
        raw_company = fields.get("company", "")
        raw_date    = fields.get("date", "")
        raw_address = fields.get("address", "")
        raw_total   = fields.get("total", "")

        # COMPANY: strip legal IDs and trailing junk (TOTAL ITEM, CHANGE, etc.)
        company = clean_company(raw_company)

        # DATE: extract a single date token from span; fallback to any date in the whole text
        date = extract_best_date(raw_date)
        if not date:
            date = extract_best_date(" ".join(orig_words))

        # TOTAL: prefer number inside span; otherwise pick from all text with keyword heuristics
        total = soft_total_norm(raw_total)
        if not total:
            # If you later store per-line strings in the dataset, pass that list here
            total = pick_total_from_lines([" ".join(orig_words)])

        # ADDRESS: just collapse spaces (your fuzzy score is already excellent)
        address = norm_spaces(raw_address)

        fields_norm = {
            "company": company,
            "date":    date,
            "address": address,
            "total":   total,
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
    ap.add_argument("--batch_size", type=int, default=2)   # kept for CLI compatibility
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--processor_dir", default=None)
    args = ap.parse_args()
    main(args)
