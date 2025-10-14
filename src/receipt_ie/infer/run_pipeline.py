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

"""
run_pipeline.py
---------------
End-to-end pipeline for intelligent receipt understanding.

Steps:
1️⃣ Use LayoutLMv3 (Part 1) to extract key fields (company, date, address, total).
2️⃣ Use GPT-based Reasoning Agent (Part 2) to refine and validate those fields.
3️⃣ Return a final JSON with an additional `agent_comment` summarizing reasoning.

This version includes:
✅ Safe JSON parsing (handles malformed model output)
✅ Retry logic for Groq/API instability
✅ Strict prompt structure for better JSON adherence
✅ Deterministic output (temperature=0)
"""

import os
import re
import json
import time
import argparse
from typing import Dict, Any

from ..utils.llm_client import LLMClient
from ..utils.postproc import (
    clean_company, soft_addr_norm, norm_date, soft_total_norm
)
from .predict_layoutlmv3 import run_inference_single


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

    # Try to extract the first JSON-like block
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidate = match.group(0)
        # Minimal repair: remove trailing commas and fix quotes
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = candidate.replace("\n", " ").replace("\r", "")
        try:
            return json.loads(candidate)
        except Exception:
            return {}
    return {}


# ------------------------------ Prompt ------------------------------ #
def build_reasoning_prompt(ocr_text: str, extracted: Dict[str, Any]) -> str:
    """
    Schema-oriented prompt enforcing strict JSON output with exact keys.
    This prompt works reliably with GPT and open models like Groq.
    
    """

    few_shot_examples = """
### EXAMPLE 1
OCR_TEXT:
KEDAI PAPAN YEW CHUAN
(0005583085-K)
LOT 276 JALAN BANTING
43800 DENGKIL, SELANGOR
TEL : 03-87686092
FAX : 03-87686092
GST ID : 000781500416
TAX INVOICE
BILL TO : SUCI ALAM JAYA TRANSPORT
NO 1 JALAN AMAN 2
TAMAN DESA 43800
DENGKIL SELANGOR
DOC NO.: CS00012013
DATE: 20/04/2018
TIME: 16:03:00
TOTAL SALES (INCLUSIVE OF GST): 87.45
CASH: 87.45
CHANGE: 0.00
GOODS SOLD ARE NOT RETURNABLE, THANK YOU

EXTRACTED_JSON:
{
  "company": "",
  "date": "",
  "address": "",
  "total": ""
}

EXPECTED_OUTPUT:
{
  "company": "KEDAI PAPAN YEW CHUAN",
  "date": "20/04/2018",
  "address": "LOT 276 JALAN BANTING 43800 DENGKIL, SELANGOR",
  "total": "87.45",
  "agent_comment": "All fields were inferred directly from clear OCR lines."
}

---

### EXAMPLE 2
OCR_TEXT:
McDonald's Bukit Bintang
TAX INVOICE
Date: 02/01/2020
Shop 12, Main Street, Kuala Lumpur
TOTAL: RM 15.90
Thank you and please come again!

EXTRACTED_JSON:
{
  "company": "",
  "date": "02/01/2020",
  "address": "Shop 12, Main Street",
  "total": ""
}

EXPECTED_OUTPUT:
{
  "company": "McDonald's Bukit Bintang",
  "date": "02/01/2020",
  "address": "Shop 12, Main Street, Kuala Lumpur",
  "total": "15.90",
  "agent_comment": "Company inferred from top OCR line containing 'McDonald's'; total value derived from 'TOTAL RM' pattern."
}
""".strip()


    return f"""
You are a receipt reasoning AI.

INPUTS:
1) OCR_TEXT: raw text from the receipt.
2) EXTRACTED_JSON: the model's initial output.

YOUR TASK:
- Review EXTRACTED_JSON in light of the OCR_TEXT.
- Correct or fill missing fields using clues in OCR_TEXT.
- Ensure JSON keys: company, date, address, total, agent_comment.
- Do NOT add extra keys, text, or markdown.

RULES:
- "company": business/vendor name (often near top).
- "date": use a plausible format (DD/MM/YYYY or DD MMM YYYY).
- "address": store or branch location.
- "total": largest numeric value near TOTAL, AMOUNT, RM, MYR, or $.
- "agent_comment": 1–2 sentences summarizing corrections (if any).
- If uncertain, leave the field blank.

Return ONLY valid JSON with these exact keys.
Follow the structure of the examples below.

{few_shot_examples}

---
### NEW RECEIPT TO PROCESS

OCR_TEXT:
{ocr_text.strip()}

EXTRACTED_JSON:
{json.dumps(extracted, ensure_ascii=False, indent=2)}

FINAL JSON OUTPUT:
""".strip()


# ------------------------------ Normalization ------------------------------ #
def normalize_refined_output(refined: Dict[str, Any]) -> Dict[str, str]:
    """Normalize all fields for consistency with evaluation pipeline."""
    out = {
        "company": clean_company(refined.get("company", "")),
        "date": norm_date(refined.get("date", "")),
        "address": soft_addr_norm(refined.get("address", "")),
        "total": soft_total_norm(refined.get("total", "")),
    }
    # Ensure a comment always exists
    comment = refined.get("agent_comment", "")
    if not isinstance(comment, str):
        comment = str(comment)
    comment = comment.strip() or "No notable corrections were required."
    out["agent_comment"] = comment
    return out


# ------------------------------ Main Pipeline ------------------------------ #
def run_pipeline_single(
    image_path: str,
    model_dir: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4-turbo",
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. LayoutLMv3 extraction
      2. LLM reasoning & validation
      3. Normalized JSON output
    """
    print(f"🔍 Processing: {os.path.basename(image_path)}")

    # Step 1 — LayoutLMv3 extraction
    extracted, ocr_text = run_inference_single(image_path, model_dir)
    print("✅ Step 1 complete: LayoutLMv3 extracted raw fields.")

    # Step 2 — Generative refinement
    llm = LLMClient(provider=llm_provider, model=llm_model, temperature=0.0)
    prompt = build_reasoning_prompt(ocr_text, extracted)

    refined = None
    for attempt in range(3):  # retry up to 3 times for reliability
        try:
            response = llm.generate(prompt)
            refined = safe_json_loads(response)
            if refined and isinstance(refined, dict):
                break
            else:
                print(f"⚠️ Attempt {attempt+1}: invalid or empty response.")
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    if not refined or not isinstance(refined, dict):
        refined = {**extracted, "agent_comment": "LLM fallback — invalid or empty response."}

    final_output = normalize_refined_output(refined)
    print("✅ Step 2 complete: Reasoning & validation finished.")
    return final_output


# ------------------------------ CLI ------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="End-to-end receipt understanding pipeline.")
    parser.add_argument("--image_path", required=True, help="Path to receipt image")
    parser.add_argument("--model_dir", required=True, help="Directory of fine-tuned LayoutLMv3 model")
    parser.add_argument("--out_path", required=True, help="Output JSON save path")
    parser.add_argument("--provider", default="openai", help="LLM provider: openai | huggingface | groq")
    parser.add_argument("--model", default="gpt-4-turbo", help="LLM model (e.g., gpt-4-turbo, openai/gpt-oss-120b)")
    args = parser.parse_args()

    result = run_pipeline_single(args.image_path, args.model_dir, args.provider, args.model)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Final validated receipt saved to: {args.out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
