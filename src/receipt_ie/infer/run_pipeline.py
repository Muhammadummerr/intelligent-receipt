"""
run_pipeline.py
---------------
End-to-end pipeline for intelligent receipt understanding.

Steps:
1️⃣ Use LayoutLMv3 (Part 1) to extract key fields (company, date, address, total).
2️⃣ Use GPT-based Reasoning Agent (Part 2) to refine and validate those fields.
3️⃣ Return a final JSON with an additional `agent_comment` summarizing reasoning.

Example Output:
{
  "company": "McDonald's",
  "date": "02/01/2020",
  "address": "Shop 12, Main Street",
  "total": "15.90",
  "agent_comment": "The company name was inferred from context as the OCR text contained 'McDonald’s' logo reference."
}
"""

import os
import json
import argparse
from typing import Dict, Any

from ..utils.llm_client import LLMClient
from ..utils.postproc import (
    clean_company, soft_addr_norm, norm_date, soft_total_norm, norm_spaces
)
from .predict_layoutlmv3 import run_inference_single  # ⚙️ Must return extracted dict + OCR text


def build_reasoning_prompt(ocr_text: str, extracted: Dict[str, Any]) -> str:
    """
    Prompt that asks GPT to correct, validate, and explain its reasoning.
    """
    return f"""
You are a document reasoning assistant for scanned receipts.
You will be given:
1. OCR text extracted from the receipt.
2. A preliminary JSON with extracted fields.

Your job:
- Correct any mistakes or missing fields.
- Ensure all 4 keys are present: company, date, address, total.
- Return a valid JSON.
- Add an extra key "agent_comment" summarizing what you inferred or corrected (1–2 sentences).

OCR TEXT:
{ocr_text.strip()}

EXTRACTED JSON:
{json.dumps(extracted, indent=2)}

Now return the corrected JSON with an additional field "agent_comment":
""".strip()


def normalize_refined_output(refined: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize GPT-refined fields for consistency with our evaluation pipeline.
    """
    return {
        "company": clean_company(refined.get("company", "")),
        "date": norm_date(refined.get("date", "")),
        "address": soft_addr_norm(refined.get("address", "")),
        "total": soft_total_norm(refined.get("total", "")),
        "agent_comment": refined.get("agent_comment", "").strip()
    }


def run_pipeline_single(image_path: str, model_dir: str, llm_provider="openai", llm_model="gpt-4-turbo") -> Dict[str, Any]:
    """
    Run the full pipeline on a single receipt image.
    """
    print(f"🔍 Processing: {os.path.basename(image_path)}")

    # Step 1 — Extract fields using LayoutLMv3
    extracted, ocr_text = run_inference_single(image_path, model_dir)
    print("✅ Step 1 complete: Extracted fields from LayoutLMv3")

    # Step 2 — Refine using LLM Reasoning
    llm = LLMClient(provider=llm_provider, model=llm_model)
    prompt = build_reasoning_prompt(ocr_text, extracted)

    try:
        response = llm.generate(prompt)
        refined = json.loads(response)
    except Exception as e:
        print(f"⚠️ LLM failed or invalid JSON: {e}")
        refined = {**extracted, "agent_comment": "LLM fallback — could not refine due to error."}

    # Step 3 — Normalize and return
    final_output = normalize_refined_output(refined)
    print("✅ Step 2 complete: Reasoning and validation done.")
    return final_output


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end receipt understanding pipeline.")
    parser.add_argument("--image_path", required=True, help="Path to a single receipt image")
    parser.add_argument("--model_dir", required=True, help="Directory of fine-tuned LayoutLMv3 model")
    parser.add_argument("--out_path", required=True, help="Path to save final refined JSON")
    parser.add_argument("--provider", default="openai", help="LLM provider (default: openai)")
    parser.add_argument("--model", default="gpt-4-turbo", help="LLM model name (default: gpt-4-turbo)")
    args = parser.parse_args()

    result = run_pipeline_single(args.image_path, args.model_dir, args.provider, args.model)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Final validated receipt saved to: {args.out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
