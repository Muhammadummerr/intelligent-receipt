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
1️⃣ LayoutLMv3 (Part 1) extracts key fields (company, date, address, total).
2️⃣ LLM Reasoning Agent (Part 2) validates, corrects, and adds agent_comment.
3️⃣ Returns normalized JSON ready for integration or evaluation.

This version includes:
✅ Safe JSON parsing (handles malformed outputs)
✅ Retry logic for Groq/API instability
✅ Auto processor fallback (fixes OSError)
✅ Strict JSON schema prompt
✅ Deterministic output (temperature=0)
"""

import os
import re
import json
import time
import argparse
from typing import Dict, Any

from ..utils.llm_client import LLMClient
from ..utils.postproc import clean_company, soft_addr_norm, norm_date, soft_total_norm
from .predict_layoutlmv3 import run_inference_single


# ------------------------------ Safe JSON Loader ------------------------------ #
def safe_json_loads(text: str) -> Dict[str, Any]:
    """Safely parse LLM outputs into JSON."""
    if not isinstance(text, str):
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
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
    Includes few-shot examples for open models like Groq.
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
DATE: 20/04/2018
TOTAL SALES (INCLUSIVE OF GST): 87.45

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
  "agent_comment": "All fields inferred directly from OCR lines."
}

---

### EXAMPLE 2
OCR_TEXT:
McDonald's Bukit Bintang
TAX INVOICE
Date: 02/01/2020
Shop 12, Main Street, Kuala Lumpur
TOTAL: RM 15.90

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
  "agent_comment": "Company inferred from OCR context; total extracted from 'TOTAL RM' line."
}
""".strip()

    return f"""
You are a receipt reasoning AI.

INPUTS:
1) OCR_TEXT: raw text from the receipt.
2) EXTRACTED_JSON: preliminary structured output.

TASK:
- Correct or fill missing fields using clues in OCR_TEXT.
- Ensure the output JSON has EXACT keys:
  ["company", "date", "address", "total", "agent_comment"].
- Do NOT include explanations or markdown outside the JSON.

RULES:
- company → vendor name (usually top).
- date → main receipt date (DD/MM/YYYY).
- address → store or branch location.
- total → largest numeric value near TOTAL / AMOUNT / RM.
- agent_comment → one short sentence describing your correction.

Follow the format of the examples below.

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
    """Normalize all LLM outputs and guarantee a valid agent_comment."""
    out = {
        "company": clean_company(refined.get("company", "")),
        "date": norm_date(refined.get("date", "")),
        "address": soft_addr_norm(refined.get("address", "")),
        "total": soft_total_norm(refined.get("total", "")),
    }
    comment = refined.get("agent_comment", "")
    if not isinstance(comment, str):
        comment = str(comment)
    out["agent_comment"] = comment.strip() or "No notable corrections required."
    return out


# ------------------------------ Pipeline Core ------------------------------ #
def run_pipeline_single(image_path: str, model_dir: str, llm_provider="openai", llm_model="gpt-4-turbo") -> Dict[str, Any]:
    """Run the full pipeline on a single receipt."""
    print(f"🔍 Processing: {os.path.basename(image_path)}")

    # Step 1 — LayoutLMv3 inference (with auto processor fallback)
    try:
        extracted, ocr_text = run_inference_single(image_path, model_dir)
    except Exception as e:
        print(f"⚠️ Processor loading failed: {e}")
        print("🔁 Loading base LayoutLMv3 processor as fallback.")
        from transformers import LayoutLMv3Processor
        base_proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        base_proc.save_pretrained(model_dir)
        extracted, ocr_text = run_inference_single(image_path, model_dir)
    print("✅ Step 1 complete: Extracted raw fields.")

    # Step 2 — Reasoning with LLM
    llm = LLMClient(provider=llm_provider, model=llm_model, temperature=0.0)
    prompt = build_reasoning_prompt(ocr_text, extracted)

    refined = None
    for attempt in range(3):
        try:
            response = llm.generate(prompt)
            refined = safe_json_loads(response)
            if refined and isinstance(refined, dict):
                break
            print(f"⚠️ Attempt {attempt+1}: invalid LLM response.")
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    if not refined:
        refined = {**extracted, "agent_comment": "LLM fallback — invalid output."}

    final_output = normalize_refined_output(refined)
    print("✅ Step 2 complete: Reasoning and validation finished.")
    return final_output


# ------------------------------ CLI ------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="End-to-end receipt understanding pipeline.")
    parser.add_argument("--image_path", required=True, help="Path to receipt image")
    parser.add_argument("--model_dir", required=True, help="Fine-tuned LayoutLMv3 model directory")
    parser.add_argument("--out_path", required=True, help="Path to save final JSON")
    parser.add_argument("--provider", default="openai", help="LLM provider: openai | huggingface | groq")
    parser.add_argument("--model", default="gpt-4-turbo", help="LLM model name")
    args = parser.parse_args()

    result = run_pipeline_single(args.image_path, args.model_dir, args.provider, args.model)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Final validated receipt saved to: {args.out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
