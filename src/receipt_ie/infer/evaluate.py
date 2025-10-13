import os, json, argparse
from typing import Dict, Optional, List
from rapidfuzz import fuzz
from ..postproc import (
    clean_company,
    soft_date_norm,
    soft_total_norm,
    soft_addr_norm,
    norm_spaces,
)

PRED_EXTS = [".json", ".txt"]
GT_EXTS = [".json", ".txt"]


def _read_json_forgiving(path: str) -> Dict:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with open(path, "rb") as f:
        return json.loads(f.read().decode("latin-1", errors="ignore"))


def _find_with_ext(dirpath: str, stem: str, exts: List[str]) -> Optional[str]:
    for e in exts:
        p = os.path.join(dirpath, stem + e)
        if os.path.isfile(p):
            return p
    return None


def load_gt(ent_dir: str, stem: str) -> Dict[str, str]:
    ent_path = _find_with_ext(ent_dir, stem, GT_EXTS)
    if ent_path is None:
        raise FileNotFoundError(f"Missing GT for {stem} in {ent_dir}")
    j = _read_json_forgiving(ent_path)

    def pick(keys, default=""):
        for k in keys:
            if k in j and j[k]:
                s = str(j[k]).strip()
                if s:
                    return s
        return default

    return {
        "company": pick(["company", "vendor", "merchant", "store", "company_name"]),
        "date": pick(["date", "invoice_date", "receipt_date"]),
        "address": pick(["address", "addr", "location", "address1", "address_1"]),
        "total": pick(["total", "amount", "grand_total", "total_amount", "total_sales", "total_sale"]),
    }


def main(args):
    pred_dir = args.pred_dir
    data_root = args.data_root
    gt_dir = os.path.join(data_root, "test", "entities")

    stems = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(pred_dir)
        if os.path.splitext(f)[1] in PRED_EXTS
    ])

    if not stems:
        print(f"No predictions found in {pred_dir}")
        return

    em_counts = {"company": 0, "date": 0, "address": 0, "total": 0}
    fuzzy_scores = {"company": [], "date": [], "address": [], "total": []}
    n = 0

    for s in stems:
        try:
            p = _read_json_forgiving(_find_with_ext(pred_dir, s, PRED_EXTS))
            g = load_gt(gt_dir, s)
        except Exception as e:
            print(f"⚠️ Skipping {s}: {e}")
            continue

        # --- Normalize both prediction & GT using same helpers ---
        p_norm = {
            "company": clean_company(p.get("company", "")),
            "date": soft_date_norm(p.get("date", "")),
            "address": soft_addr_norm(p.get("address", "")),
            "total": soft_total_norm(p.get("total", "")),
        }
        g_norm = {
            "company": clean_company(g.get("company", "")),
            "date": soft_date_norm(g.get("date", "")),
            "address": soft_addr_norm(g.get("address", "")),
            "total": soft_total_norm(g.get("total", "")),
        }

        # --- Exact / fuzzy match counting ---
        for k in em_counts.keys():
            score = fuzz.token_set_ratio(p_norm[k], g_norm[k])
            if k == "address":
                if score >= 85:
                    em_counts[k] += 1
            else:
                if score >= 95:
                    em_counts[k] += 1

            if g_norm[k] or p_norm[k]:
                fuzzy_scores[k].append(score)

        n += 1

    def pct(x): return 100.0 * x / max(1, n)
    print(f"\n📊 Evaluation Summary ({n} receipts)\n")
    print("=== Exact-Match (Fuzzy ≥ thresholds) ===")
    for k in em_counts:
        print(f"  {k:<8}: {em_counts[k]:>4}/{n:<4} ({pct(em_counts[k]):5.1f}%)")

    print("\n=== Average Fuzzy Similarity (token_set_ratio) ===")
    for k, vals in fuzzy_scores.items():
        avg = sum(vals)/len(vals) if vals else 0.0
        print(f"  {k:<8}: {avg:5.1f}")

    all_fuzzy = sum(sum(v) for v in fuzzy_scores.values())
    all_counts = sum(len(v) for v in fuzzy_scores.values())
    avg_fuzzy = all_fuzzy / max(1, all_counts)
    print(f"\nOverall fuzzy mean similarity: {avg_fuzzy:5.1f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()
    main(args)
