# src/receipt_ie/infer/evaluate.py
import os, json, argparse, re
from typing import Dict
from rapidfuzz import fuzz
from ..utils.postproc import norm_spaces, norm_date, norm_total

def load_gt(ent_path: str) -> Dict[str,str]:
    with open(ent_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    # robust to minor key variants
    def pick(j, keys, default=""):
        for k in keys:
            if k in j and j[k]:
                s = str(j[k]).strip()
                if s:
                    return s
        return default
    return {
        "company": pick(j, ["company","vendor","merchant","store","company_name"]),
        "date":    pick(j, ["date","invoice_date","receipt_date"]),
        "address": pick(j, ["address","addr","location","address1","address_1"]),
        "total":   pick(j, ["total","amount","grand_total","total_amount","total_sales","total_sale"]),
    }

def main(args):
    pred_dir = args.pred_dir
    data_root = args.data_root
    gt_dir = os.path.join(data_root, "test", "entities")

    stems = [os.path.splitext(f)[0] for f in os.listdir(pred_dir) if f.endswith(".json")]
    stems.sort()

    n = 0
    em_counts = {"company":0,"date":0,"address":0,"total":0}
    fuzzy_addr = []

    for s in stems:
        with open(os.path.join(pred_dir, s+".json"), "r", encoding="utf-8") as f:
            p = json.load(f)
        g = load_gt(os.path.join(gt_dir, s+".json"))

        # normalize
        p_norm = {
            "company": norm_spaces(p.get("company","")),
            "date":    norm_date(p.get("date","")),
            "address": norm_spaces(p.get("address","")),
            "total":   norm_total(p.get("total","")),
        }
        g_norm = {
            "company": norm_spaces(g.get("company","")),
            "date":    norm_date(g.get("date","")),
            "address": norm_spaces(g.get("address","")),
            "total":   norm_total(g.get("total","")),
        }

        # exact match stats
        for k in em_counts.keys():
            if p_norm[k] == g_norm[k]:
                em_counts[k] += 1

        # address fuzzy score (token-set)
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
