"""
compare_with_agent.py
---------------------
Compare the original model predictions vs. LLM-refined reasoning agent outputs.

Metrics:
- Exact match rate (with fuzzy threshold for address)
- Average fuzzy similarity (token_set_ratio)
- Field-wise improvement deltas
"""

import os
import json
import argparse
from typing import Dict
from rapidfuzz import fuzz
from ..utils.postproc import clean_company, soft_addr_norm, norm_date, soft_total_norm


def read_json_forgiving(path: str) -> Dict:
    """Read JSON robustly with multiple encodings."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    with open(path, "rb") as f:
        try:
            return json.loads(f.read().decode("latin-1", errors="ignore"))
        except Exception:
            return {}
    

def normalize_fields(j: Dict) -> Dict[str, str]:
    """Apply consistent normalization across both sources."""
    return {
        "company": clean_company(j.get("company", "")),
        "date": norm_date(j.get("date", "")),
        "address": soft_addr_norm(j.get("address", "")),
        "total": soft_total_norm(j.get("total", "")),
    }


def evaluate_pair(pred_json: Dict, gt_json: Dict) -> Dict[str, float]:
    """Compute metrics for one prediction vs. ground truth."""
    metrics = {}
    for k in ["company", "date", "address", "total"]:
        p, g = pred_json.get(k, ""), gt_json.get(k, "")
        if not (p or g):
            metrics[k] = {"em": 0.0, "fuzzy": 0.0}
            continue
        fuzzy = fuzz.token_set_ratio(p, g)
        if k == "address":
            em = 1.0 if fuzzy >= 85 else 0.0
        else:
            em = 1.0 if p == g else 0.0
        metrics[k] = {"em": em, "fuzzy": fuzzy}
    return metrics


def aggregate_results(metrics_list):
    """Compute averages across all receipts."""
    agg = {k: {"em": 0.0, "fuzzy": 0.0} for k in ["company", "date", "address", "total"]}
    n = len(metrics_list)
    if n == 0:
        return agg
    for m in metrics_list:
        for k in agg:
            agg[k]["em"] += m[k]["em"]
            agg[k]["fuzzy"] += m[k]["fuzzy"]
    for k in agg:
        agg[k]["em"] = (agg[k]["em"] / n) * 100
        agg[k]["fuzzy"] = (agg[k]["fuzzy"] / n)
    return agg


def main(args):
    pred_dir = args.pred_dir
    refined_dir = args.refined_dir
    gt_dir = args.gt_dir

    stems = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(gt_dir)
        if f.lower().endswith((".json", ".txt"))
    ])
    if not stems:
        print(f"⚠️ No ground truth files in {gt_dir}")
        return

    before_metrics, after_metrics = [], []

    for s in stems:
        pred_path = os.path.join(pred_dir, s + ".json")
        refined_path = os.path.join(refined_dir, s + ".json")
        gt_path = os.path.join(gt_dir, s + ".txt")

        if not (os.path.isfile(pred_path) and os.path.isfile(refined_path) and os.path.isfile(gt_path)):
            continue

        pred_raw = normalize_fields(read_json_forgiving(pred_path))
        refined_raw = normalize_fields(read_json_forgiving(refined_path))
        gt_raw = normalize_fields(read_json_forgiving(gt_path))

        before_metrics.append(evaluate_pair(pred_raw, gt_raw))
        after_metrics.append(evaluate_pair(refined_raw, gt_raw))

    agg_before = aggregate_results(before_metrics)
    agg_after = aggregate_results(after_metrics)

    print("\n📊 Comparison Summary\n")
    print(f"{'Field':<10} {'Before(EM%)':>12} {'After(EM%)':>12} {'ΔEM':>8}   {'Before(Fuzzy)':>15} {'After(Fuzzy)':>15} {'ΔFuzzy':>8}")
    print("-" * 78)

    for k in ["company", "date", "address", "total"]:
        delta_em = agg_after[k]["em"] - agg_before[k]["em"]
        delta_fuzzy = agg_after[k]["fuzzy"] - agg_before[k]["fuzzy"]
        print(f"{k:<10} {agg_before[k]['em']:>10.1f}% {agg_after[k]['em']:>10.1f}% {delta_em:>+7.1f}   "
              f"{agg_before[k]['fuzzy']:>13.1f} {agg_after[k]['fuzzy']:>13.1f} {delta_fuzzy:>+7.1f}")

    print("-" * 78)
    overall_fuzzy_before = sum(v["fuzzy"] for v in agg_before.values()) / 4
    overall_fuzzy_after = sum(v["fuzzy"] for v in agg_after.values()) / 4
    print(f"\n⭐ Overall fuzzy mean improvement: {overall_fuzzy_after - overall_fuzzy_before:+.2f}")
    print(f"✅ Total receipts evaluated: {len(stems)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare raw vs. LLM-refined receipt predictions.")
    ap.add_argument("--pred_dir", required=True, help="Directory of original LayoutLMv3 predictions")
    ap.add_argument("--refined_dir", required=True, help="Directory of LLM reasoning agent outputs")
    ap.add_argument("--gt_dir", required=True, help="Directory of ground-truth JSONs")
    args = ap.parse_args()
    main(args)
