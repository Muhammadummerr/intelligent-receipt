import os, json, re
from collections import defaultdict
from pathlib import Path
from rapidfuzz import fuzz

# --- Import project helpers ---
from receipt_ie.utils.postproc import (
    clean_company,
    soft_date_norm,
    soft_total_norm,
    soft_addr_norm,
)
from receipt_ie.data.align import load_entities

# Paths
PRED_DIR = "/kaggle/working/preds"
GT_DIR = "/kaggle/input/receipt-dataset/test/entities"
OUT_FILE = "/kaggle/working/mismatches.txt"

# --- encoding-tolerant JSON reader ---
def read_json_any(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with open(path, "rb") as f:
        return json.loads(f.read().decode("latin-1", errors="ignore"))

def find_with_ext(dirpath, stem, exts=(".json", ".txt")):
    for e in exts:
        p = os.path.join(dirpath, stem + e)
        if os.path.isfile(p):
            return p
    return None

def load_gt(stem):
    path = find_with_ext(GT_DIR, stem, (".json", ".txt"))
    if path is None:
        return None
    ent = load_entities(path)
    return {
        "company": ent.company,
        "date": ent.date,
        "address": ent.address,
        "total": ent.total,
    }

# --- Collect stems ---
stems = sorted(
    Path(f).stem
    for f in os.listdir(PRED_DIR)
    if os.path.splitext(f)[1] in (".json", ".txt")
)
print("Pred files:", len(stems))

pairs = []
for s in stems:
    pred_path = find_with_ext(PRED_DIR, s)
    if not pred_path:
        continue
    pred = read_json_any(pred_path)
    gt = load_gt(s)
    if gt is None:
        continue

    pairs.append({
        "stem": s,
        "pred_raw": {
            "company": pred.get("company", ""),
            "date": pred.get("date", ""),
            "address": pred.get("address", ""),
            "total": pred.get("total", ""),
        },
        "gt_raw": gt,
        "pred_norm": {
            "company": clean_company(pred.get("company", "")),
            "date": soft_date_norm(pred.get("date", "")),
            "address": soft_addr_norm(pred.get("address", "")),
            "total": soft_total_norm(pred.get("total", "")),
        },
        "gt_norm": {
            "company": clean_company(gt.get("company", "")),
            "date": soft_date_norm(gt.get("date", "")),
            "address": soft_addr_norm(gt.get("address", "")),
            "total": soft_total_norm(gt.get("total", "")),
        },
    })

print("Aligned pairs:", len(pairs))

# --- Find mismatches (fuzzy for address) ---
mismatch_by_field = defaultdict(list)
for r in pairs:
    for k in ("company", "date", "address", "total"):
        if k == "address":
            score = fuzz.token_set_ratio(r["pred_norm"][k], r["gt_norm"][k])
            if score < 85:
                mismatch_by_field[k].append((r, score))
        else:
            if r["pred_norm"][k] != r["gt_norm"][k]:
                score = fuzz.token_set_ratio(r["pred_norm"][k], r["gt_norm"][k])
                mismatch_by_field[k].append((r, score))

# --- Save all mismatches ---
with open(OUT_FILE, "w", encoding="utf-8") as f:
    total_items = len(pairs)
    f.write(f"📊 Mismatch Report ({total_items} receipts)\n\n")

    for field, items in mismatch_by_field.items():
        f.write(f"=== {field.upper()} mismatches ({len(items)}) ===\n")
        for r, score in items:
            f.write(f"\n- {r['stem']}\n")
            f.write(f"  Fuzzy similarity: {score:.1f}\n")
            f.write(f"  PRED raw : {r['pred_raw'][field]}\n")
            f.write(f"  GT   raw : {r['gt_raw'][field]}\n")
            f.write(f"  PRED norm: {r['pred_norm'][field]}\n")
            f.write(f"  GT   norm: {r['gt_norm'][field]}\n")
            f.write("\n")

print(f"\n✅ Saved mismatch report to: {OUT_FILE}")
print("You can now share this text file to analyze where the model goes wrong.")
