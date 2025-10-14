# """
# run_pipeline.py
# ---------------
# End-to-end pipeline for intelligent receipt understanding.

# Steps:
# 1️⃣ Use LayoutLMv3 (Part 1) to extract key fields (company, date, address, total).
# 2️⃣ Use GPT-based Reasoning Agent (Part 2) to refine and validate those fields.
# 3️⃣ Return a final JSON with an additional `agent_comment` summarizing reasoning.

# Example Output:
# {
#   "company": "McDonald's",
#   "date": "02/01/2020",
#   "address": "Shop 12, Main Street",
#   "total": "15.90",
#   "agent_comment": "The company name was inferred from context as the OCR text contained 'McDonald’s' logo reference."
# }
# """

# import os
# import json
# import argparse
# from typing import Dict, Any

# from ..utils.llm_client import LLMClient
# from ..utils.postproc import (
#     clean_company, soft_addr_norm, norm_date, soft_total_norm, norm_spaces
# )
# from .predict_layoutlmv3 import run_inference_single  # ⚙️ Must return extracted dict + OCR text


# def build_reasoning_prompt(ocr_text: str, extracted: Dict[str, Any]) -> str:
#     """
#     Prompt that asks GPT to correct, validate, and explain its reasoning.
#     """
#     return f"""
# You are a document reasoning assistant for scanned receipts.
# You will be given:
# 1. OCR text extracted from the receipt.
# 2. A preliminary JSON with extracted fields.

# Your job:
# - Correct any mistakes or missing fields.
# - Ensure all 4 keys are present: company, date, address, total.
# - Return a valid JSON.
# - Add an extra key "agent_comment" summarizing what you inferred or corrected (1–2 sentences).

# OCR TEXT:
# {ocr_text.strip()}

# EXTRACTED JSON:
# {json.dumps(extracted, indent=2)}

# Now return the corrected JSON with an additional field "agent_comment":
# """.strip()


# def normalize_refined_output(refined: Dict[str, Any]) -> Dict[str, str]:
#     """
#     Normalize GPT-refined fields for consistency with our evaluation pipeline.
#     """
#     return {
#         "company": clean_company(refined.get("company", "")),
#         "date": norm_date(refined.get("date", "")),
#         "address": soft_addr_norm(refined.get("address", "")),
#         "total": soft_total_norm(refined.get("total", "")),
#         "agent_comment": refined.get("agent_comment", "").strip()
#     }


# def run_pipeline_single(image_path: str, model_dir: str, llm_provider="openai", llm_model="gpt-4-turbo") -> Dict[str, Any]:
#     """
#     Run the full pipeline on a single receipt image.
#     """
#     print(f"🔍 Processing: {os.path.basename(image_path)}")

#     # Step 1 — Extract fields using LayoutLMv3
#     extracted, ocr_text = run_inference_single(image_path, model_dir)
#     print("✅ Step 1 complete: Extracted fields from LayoutLMv3")

#     # Step 2 — Refine using LLM Reasoning
#     llm = LLMClient(provider=llm_provider, model=llm_model)
#     prompt = build_reasoning_prompt(ocr_text, extracted)

#     try:
#         response = llm.generate(prompt)
#         refined = json.loads(response)
#     except Exception as e:
#         print(f"⚠️ LLM failed or invalid JSON: {e}")
#         refined = {**extracted, "agent_comment": "LLM fallback — could not refine due to error."}

#     # Step 3 — Normalize and return
#     final_output = normalize_refined_output(refined)
#     print("✅ Step 2 complete: Reasoning and validation done.")
#     return final_output


# def main():
#     parser = argparse.ArgumentParser(description="Run end-to-end receipt understanding pipeline.")
#     parser.add_argument("--image_path", required=True, help="Path to a single receipt image")
#     parser.add_argument("--model_dir", required=True, help="Directory of fine-tuned LayoutLMv3 model")
#     parser.add_argument("--out_path", required=True, help="Path to save final refined JSON")
#     parser.add_argument("--provider", default="openai", help="LLM provider (default: openai)")
#     parser.add_argument("--model", default="gpt-4-turbo", help="LLM model name (default: gpt-4-turbo)")
#     args = parser.parse_args()

#     result = run_pipeline_single(args.image_path, args.model_dir, args.provider, args.model)
#     os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

#     with open(args.out_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)

#     print(f"\n🎉 Final validated receipt saved to: {args.out_path}")
#     print(json.dumps(result, indent=2))


# if __name__ == "__main__":
#     main()


"""
run_pipeline.py
---------------
End-to-end pipeline for intelligent receipt understanding.

Steps:
1) LayoutLMv3 (Part 1) extracts preliminary fields (company, date, address, total).
2) LLM Reasoning Agent (Part 2) validates, corrects, and adds a short agent_comment.
3) Outputs normalized JSON ready for Part 3 evaluation / demo.

This file is intentionally robust:
- Strict, schema-like prompt
- Safe JSON parsing (recovers from common LLM formatting hiccups)
- Always returns agent_comment
"""

import os
import re
import json
import argparse
from typing import Dict, Any

from ..utils.llm_client import LLMClient
from ..utils.postproc import (
    clean_company, soft_addr_norm, norm_date, soft_total_norm
)
from .predict_layoutlmv3 import run_inference_single  # returns (extracted_dict, ocr_text)


# ------------------------------ Utilities ------------------------------ #
def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse JSON safely:
      - first try direct json.loads
      - if it fails, extract the largest {...} block and try again
      - if still fails, return {}
    """
    if not isinstance(text, str):
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to recover the first JSON object span
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            # Try a minimal repair: remove trailing commas
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except Exception:
                return {}

    return {}


# ------------------------------ Prompt ------------------------------ #
def build_reasoning_prompt(ocr_text: str, extracted: Dict[str, Any]) -> str:
    """
    Schema-oriented prompt that forces JSON-only output with the required keys.
    Short, explicit constraints help Groq/OSS models behave.
    """
    return f"""
You are a receipt reasoning agent.

INPUTS:
1) OCR_TEXT: raw text from the receipt.
2) EXTRACTED_JSON: preliminary fields from a model.

TASK:
- Validate or correct these fields using OCR_TEXT.
- If uncertain, leave the field empty (do NOT invent values).
- Output JSON ONLY (no markdown, no prose) with EXACT keys:
  company (string), date (string), address (string), total (string), agent_comment (string)

CONSTRAINTS:
- date: use a plausible receipt date (e.g., DD/MM/YYYY, D/M/YYYY, DD MMM YYYY). If multiple dates exist, choose the main transaction date.
- total: choose the final amount (largest 2-decimal currency near "TOTAL", "AMOUNT", "RM", "MYR", "$"). Keep it as a string with two decimals if possible.
- company: vendor/merchant name (often near top).
- address: store/location line(s) if present.
- agent_comment: 1–2 short sentences explaining corrections or confidence.

Return ONLY valid JSON.

OCR_TEXT:
{ocr_text.strip()}

EXTRACTED_JSON:
{json.dumps(extracted, ensure_ascii=False, indent=2)}

NOW RETURN THE FINAL JSON WITH THESE EXACT KEYS:
["company","date","address","total","agent_comment"]
""".strip()


# ------------------------------ Normalization ------------------------------ #
def normalize_refined_output(refined: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize LLM-refined fields for consistency with the evaluation pipeline.
    Guarantees the agent_comment key exists.
    """
    out = {
        "company": clean_company(refined.get("company", "")),
        "date": norm_date(refined.get("date", "")),
        "address": soft_addr_norm(refined.get("address", "")),
        "total": soft_total_norm(refined.get("total", "")),
    }
    # Ensure we always provide a short comment
    comment = refined.get("agent_comment", "")
    if not isinstance(comment, str):
        comment = str(comment)
    comment = comment.strip()
    if not comment:
        comment = "No changes required or low-confidence refinement."
    out["agent_comment"] = comment
    return out


# ------------------------------ Pipeline ------------------------------ #
def run_pipeline_single(
    image_path: str,
    model_dir: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4-turbo",
) -> Dict[str, Any]:
    """
    Run the full pipeline on a single receipt image:
      1) LayoutLMv3 extraction (Part 1)
      2) LLM reasoning (Part 2)
      3) Normalized, validated JSON with agent_comment (Part 3)
    """
    print(f"🔍 Processing: {os.path.basename(image_path)}")

    # Step 1 — Extract fields using LayoutLMv3
    extracted, ocr_text = run_inference_single(image_path, model_dir)
    print("✅ Step 1 complete: Extracted fields from LayoutLMv3")

    # Step 2 — Refine using LLM Reasoning
    llm = LLMClient(provider=llm_provider, model=llm_model)
    prompt = build_reasoning_prompt(ocr_text, extracted)

    refined = {}
    try:
        response = llm.generate(prompt)
        refined = safe_json_loads(response)
    except Exception as e:
        print(f"⚠️ LLM request failed: {e}")
        refined = {}

    # If parsing yielded nothing, fall back to extracted + comment
    if not refined or not isinstance(refined, dict):
        refined = {
            **extracted,
            "agent_comment": "LLM fallback — output could not be parsed, using model extraction."
        }

    # Step 3 — Normalize and return
    final_output = normalize_refined_output(refined)
    print("✅ Step 2 complete: Reasoning and validation done.")
    return final_output


# ------------------------------ CLI ------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Run end-to-end receipt understanding pipeline.")
    parser.add_argument("--image_path", required=True, help="Path to a single receipt image")
    parser.add_argument("--model_dir", required=True, help="Directory of fine-tuned LayoutLMv3 model")
    parser.add_argument("--out_path", required=True, help="Path to save final refined JSON")
    parser.add_argument("--provider", default="openai", help="LLM provider: openai | huggingface | groq")
    parser.add_argument("--model", default="gpt-4-turbo", help="LLM model name (e.g., gpt-4-turbo, openai/gpt-oss-120b)")
    args = parser.parse_args()

    result = run_pipeline_single(
        image_path=args.image_path,
        model_dir=args.model_dir,
        llm_provider=args.provider,
        llm_model=args.model,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Final validated receipt saved to: {args.out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
