# src/receipt_ie/infer/evaluate.py
import os, json, argparse
from typing import Dict, Optional, List
from rapidfuzz import fuzz
# from ..utils.postproc import norm_spaces, norm_date, norm_total
from ..utils.postproc import norm_spaces, norm_date, norm_total, clean_company

PRED_EXTS = [".json", ".txt"]
GT_EXTS   = [".json", ".txt"]

def _read_json_forgiving(path: str) -> Dict:
    # tolerate cp1252/latin-1, etc.
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
        "company": pick(["company","vendor","merchant","store","company_name"]),
        "date":    pick(["date","invoice_date","receipt_date"]),
        "address": pick(["address","addr","location","address1","address_1"]),
        "total":   pick(["total","amount","grand_total","total_amount","total_sales","total_sale"]),
    }

def main(args):
    pred_dir = args.pred_dir
    data_root = args.data_root
    gt_dir = os.path.join(data_root, "test", "entities")

    # collect stems from predictions (accept .json or .txt)
    stems = []
    for f in os.listdir(pred_dir):
        base, ext = os.path.splitext(f)
        if ext in PRED_EXTS:
            stems.append(base)
    stems.sort()

    n = 0
    em_counts = {"company":0,"date":0,"address":0,"total":0}
    fuzzy_addr = []

    for s in stems:
        pred_path = _find_with_ext(pred_dir, s, PRED_EXTS)
        if pred_path is None:
            continue
        p = _read_json_forgiving(pred_path)
        g = load_gt(gt_dir, s)

        # normalize
        
        p_norm = {
            "company": clean_company(p.get("company","")),
            "date":    norm_date(p.get("date","")),
            "address": norm_spaces(p.get("address","")),
            "total":   norm_total(p.get("total","")),
        }
        g_norm = {
            "company": clean_company(g.get("company","")),
            "date":    norm_date(g.get("date","")),
            "address": norm_spaces(g.get("address","")),
            "total":   norm_total(g.get("total","")),
        }


        # exact match stats
        for k in em_counts.keys():
            if p_norm[k] == g_norm[k]:
                em_counts[k] += 1

        # address fuzzy score
        if g_norm["address"] or p_norm["address"]:
            fuzzy_addr.append(fuzz.token_set_ratio(p_norm["address"], g_norm["address"]))
        n += 1

    def pct(x): return 100.0 * x / max(1,n)
    print(f"Test items: {n}")
    print("Exact-match accuracy:")
    print(f"  company: {em_counts['company']}/{n} ({pct(em_counts['company']):.1f}%)")
    print(f"  date   : {em_counts['date']}/{n} ({pct(em_counts['date']):.1f}%)")
    print(f"  total  : {em_counts['total']}/{n} ({pct(em_counts['total']):.1f}%)")
    print(f"  address: {em_counts['address']}/{n} ({pct(em_counts['address']):.1f}%)")
    if fuzzy_addr:
        avg_addr = sum(fuzzy_addr)/len(fuzzy_addr)
        print(f"  address (avg RapidFuzz token_set_ratio): {avg_addr:.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()
    main(args)
