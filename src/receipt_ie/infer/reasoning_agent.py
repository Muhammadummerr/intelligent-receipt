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
    Optimized reasoning prompt for LLM-based receipt correction.
    Enforces strict JSON structure and adds self-check instructions
    to minimize parsing errors and hallucinations.
    """
    return f"""
You are a **document reasoning agent** that processes scanned receipts.

### TASK OVERVIEW
You are given:
1. **OCR text** from the receipt (raw extracted text).
2. **Partially extracted fields** in JSON form (company, date, address, total).

Your task:
- Review the OCR text and correct any incorrect or missing fields in the JSON.
- Ensure all 4 fields are logically consistent and realistic.
- Do NOT fabricate data — if a value cannot be found, leave it as an empty string ("").
- Follow Malaysian formatting conventions (e.g., dates, currency "RM").

### OUTPUT INSTRUCTIONS (MUST FOLLOW STRICTLY)
- Output **ONLY valid JSON** — no Markdown, no explanations, no commentary.
- The JSON must have exactly these 4 keys, spelled exactly:
  - company
  - date
  - address
  - total
- Each value must be a string (even if numeric).
- The total should contain the numeric amount (e.g., "32.50"), not "RM32.50".

### VALIDATION RULES
- **Company:** Usually near the top, may include "SDN BHD", "ENTERPRISE", or logo-like text.
- **Date:** Must resemble DD/MM/YYYY, DD-MM-YYYY, or DD MMM YYYY.
- **Address:** Should include street or area terms like "JALAN", "TAMAN", "KUALA LUMPUR", etc.
- **Total:** Should be the largest number near the words "TOTAL", "CASH", "AMOUNT DUE", or "RM".

### SELF-CHECK BEFORE RETURNING:
1. Ensure all 4 keys exist.
2. Ensure it is **valid JSON** (parseable by Python json.loads).
3. Do not include any text before or after the JSON (no explanations).

### OCR TEXT:
{ocr_text.strip()}

### EXTRACTED JSON:
{json.dumps(extracted, indent=2)}

Now return ONLY the corrected JSON that meets all the above rules.
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
                        provider: str = "groq", model: str = "openai/gpt-oss-120b"):
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
    parser.add_argument("--provider", default="groq", choices=["openai", "huggingface", "groq"], help="LLM backend provider")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name")
    args = parser.parse_args()

    run_reasoning_agent(args.pred_dir, args.ocr_dir, args.out_dir, args.provider, args.model)


if __name__ == "__main__":
    main()
