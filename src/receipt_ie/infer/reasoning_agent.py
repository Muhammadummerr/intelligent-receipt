"""
reasoning_agent.py
-------------------
This module implements the Generative AI reasoning agent that
validates and corrects extracted receipt fields using an LLM.

It takes the LayoutLMv3 model's predictions and OCR text as input,
and outputs refined JSONs after logical consistency reasoning.
"""

import os
import json
import argparse
from typing import Dict, Any
from ..utils.llm_client import LLMClient
from ..utils.postproc import clean_company, soft_addr_norm, norm_date, soft_total_norm, norm_spaces


def build_prompt(ocr_text: str, extracted: Dict[str, Any]) -> str:
    """
    Construct the reasoning prompt for the LLM.
    The prompt encourages structured reasoning and JSON-only output.
    """
    return f"""
You are a document reasoning assistant for receipts.
Your task is to analyze the OCR text and the extracted fields (JSON)
and return a corrected JSON with consistent values.

Follow these rules:
1. If a field seems missing or incorrect, infer the most likely value from the OCR text.
2. Always output valid JSON with keys: company, date, address, total.
3. Do NOT include explanations — only the final JSON.
4. Use currency or date context clues to infer totals and dates.

OCR TEXT:
{ocr_text.strip()}

EXTRACTED JSON:
{json.dumps(extracted, indent=2)}

Now return the corrected JSON:
""".strip()


def normalize_refined_output(refined: Dict[str, Any]) -> Dict[str, str]:
    """
    Apply standard normalization to LLM-refined outputs to
    maintain consistency with evaluation preprocessing.
    """
    return {
        "company": clean_company(refined.get("company", "")),
        "date": norm_date(refined.get("date", "")),
        "address": soft_addr_norm(refined.get("address", "")),
        "total": soft_total_norm(refined.get("total", "")),
    }


def run_reasoning_agent(pred_dir: str, ocr_dir: str, out_dir: str,
                        provider: str = "openai", model: str = "gpt-4-turbo"):
    """
    Main pipeline: load each predicted JSON, read OCR text, run reasoning,
    and save corrected JSONs to `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)
    llm = LLMClient(provider=provider, model=model)

    preds = [f for f in os.listdir(pred_dir) if f.endswith(".json")]
    if not preds:
        print(f"⚠️ No prediction files found in {pred_dir}")
        return

    for i, fname in enumerate(preds, start=1):
        stem = os.path.splitext(fname)[0]
        pred_path = os.path.join(pred_dir, fname)
        ocr_path = os.path.join(ocr_dir, stem + ".txt")
        out_path = os.path.join(out_dir, fname)

        if not os.path.isfile(pred_path) or not os.path.isfile(ocr_path):
            print(f"⚠️ Skipping {stem}: missing OCR or pred JSON")
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            extracted = json.load(f)
        with open(ocr_path, "r", encoding="utf-8", errors="ignore") as f:
            ocr_text = f.read()

        prompt = build_prompt(ocr_text, extracted)
        try:
            response = llm.generate(prompt)
            refined = json.loads(response)
        except Exception as e:
            print(f"⚠️ LLM or parsing failed for {stem}: {e}")
            refined = extracted

        normalized = normalize_refined_output(refined)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

        print(f"[{i}/{len(preds)}] ✅ Saved refined JSON → {out_path}")

    print(f"\n✅ All reasoning results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run the LLM reasoning agent on extracted receipts.")
    parser.add_argument("--pred_dir", required=True, help="Directory of LayoutLMv3 JSON predictions")
    parser.add_argument("--ocr_dir", required=True, help="Directory containing OCR text files (.txt)")
    parser.add_argument("--out_dir", required=True, help="Output directory for refined JSONs")
    parser.add_argument("--provider", default="openai", choices=["openai", "huggingface"], help="LLM backend provider")
    parser.add_argument("--model", default="gpt-4-turbo", help="Model name")
    args = parser.parse_args()

    run_reasoning_agent(args.pred_dir, args.ocr_dir, args.out_dir, args.provider, args.model)


if __name__ == "__main__":
    main()
