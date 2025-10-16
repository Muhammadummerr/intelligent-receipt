"""
run_pipeline.py
---------------
End-to-end pipeline for intelligent receipt understanding.

Steps:
1️⃣ LayoutLMv3 (Part 1) extracts key fields (company, date, address, total).
2️⃣ LLM Reasoning Agent (Part 2) validates, corrects, and adds agent_comment.
3️⃣ Returns normalized JSON ready for integration or evaluation.

Enhanced features:
✅ OCR fallback via EasyOCR (for real-world receipts)
✅ Safe JSON parsing (handles malformed outputs)
✅ Retry logic for Groq/API instability
✅ Auto processor/model fallback
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
# from .predict_layoutlmv3 import run_inference_single
from .predict_donut import run_inference_single

os.environ["HF_HOME"] = "/kaggle/temp/hf_home"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/temp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf_models"

# ------------------------------ Safe JSON Loader ------------------------------ #
def safe_json_loads(text: str) -> Dict[str, Any]:
    """Safely parse LLM outputs into JSON, handling malformed or markdown-wrapped outputs."""
    if not isinstance(text, str):
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt to extract JSON from markdown or text
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        candidate = candidate.replace("\n", " ").replace("\r", "")
        try:
            return json.loads(candidate)
        except Exception:
            return {}
    return {}


# ------------------------------ Prompt Builder ------------------------------ #
def build_reasoning_prompt(ocr_text: str, extracted: Dict[str, Any]) -> str:
    """
    Schema-oriented prompt enforcing strict JSON output with exact keys.
    Includes few-shot examples for improved reliability.
    """

    few_shot_examples = """
### EXAMPLE 1
OCR_TEXT:
KEDAI PAPAN YEW CHUAN
LOT 276 JALAN BANTING
43800 DENGKIL, SELANGOR
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
1) OCR_TEXT: raw text extracted from the receipt (may be noisy OCR).
2) EXTRACTED_JSON: preliminary structured output.

TASK:
- Correct or fill missing fields using clues in OCR_TEXT.
- Ensure output JSON has EXACT keys:
  ["company", "date", "address", "total", "agent_comment"].
- Do NOT include explanations or markdown outside the JSON.

RULES:
- company → vendor or business name (usually top).
- date → primary receipt date (DD/MM/YYYY format).
- address → store or branch address.
- total → largest numeric value near TOTAL/AMOUNT.
- agent_comment → one short sentence summarizing how corrections were made.

Follow the examples below for structure and clarity.

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
    """Normalize LLM-refined fields for consistency with evaluation."""
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
    """Run the complete pipeline for a single receipt image."""
    print(f"🔍 Processing: {os.path.basename(image_path)}")

    # Step 1 — LayoutLMv3 Extraction
    try:
        extracted, ocr_text = run_inference_single(image_path, model_dir)
    except Exception as e:
        print(f"⚠️ Initial inference failed: {e}")
        extracted, ocr_text = {"company": "", "date": "", "address": "", "total": ""}, ""

    print(f"Extracted: {extracted}")
    print("✅ Step 1 complete: Extracted raw fields from LayoutLMv3.")

    # Step 2 — Reasoning via LLM
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


# ------------------------------ CLI Entry ------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="End-to-end receipt understanding pipeline.")
    parser.add_argument("--image_path", required=True, help="Path to receipt image")
    parser.add_argument("--model_dir", required=True, help="Fine-tuned LayoutLMv3 model directory or HuggingFace repo")
    parser.add_argument("--out_path", required=True, help="Path to save final JSON")
    parser.add_argument("--provider", default="openai", help="LLM provider: openai | groq | huggingface")
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
