import json, os, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from rapidfuzz import fuzz
from collections import Counter
from .boxes import BoxLine, sort_reading_order
from ..utils.text import normalize_spaces, strip_punct_lower, is_date_like

FIELDS = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]
LABELS = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE", "I-DATE",
    "B-ADDRESS", "I-ADDRESS",
    "B-TOTAL", "I-TOTAL"
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


# ------------------------------------------------------------------
# ENTITY STRUCTURE
# ------------------------------------------------------------------
@dataclass
class EntityGT:
    company: str
    date: str
    address: str
    total: str


def load_entities(path: str) -> EntityGT:
    """Load JSON or TXT entities safely and normalize."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            j = json.load(f)
        except Exception:
            txt = f.read().strip()
            j = json.loads(txt)

    def pick(keys, default=""):
        for k in keys:
            if k in j and j[k]:
                val = str(j[k]).strip()
                if val != "":
                    return val
        return default

    company = pick(["company", "vendor", "merchant", "store", "company_name"])
    date = pick(["date", "invoice_date", "receipt_date"])
    address = pick(["address", "addr", "location", "address_1", "address1"])
    total = pick(["total", "amount", "total_amount", "grand_total", "total_sales", "total_sale"])

    return EntityGT(
        company=normalize_spaces(company),
        date=normalize_spaces(date),
        address=normalize_spaces(address),
        total=normalize_spaces(total),
    )


# ------------------------------------------------------------------
# FUZZY SCORERS
# ------------------------------------------------------------------
def score_company(line: str, gt_company: str) -> float:
    return fuzz.partial_ratio(line, gt_company)


def score_address(line: str, gt_address: str) -> float:
    return fuzz.token_set_ratio(line, gt_address)


def is_total_amount(line: str) -> Optional[float]:
    s = line or ""
    s = re.sub(r"(RM|MYR|\$|USD)\s*", "", s, flags=re.I)
    s = re.sub(r"(\d)\s+(\d)", r"\1\2", s)
    s = re.sub(r"[^\dA-Za-z\.\s]", "", s)

    if re.search(r"\b(total|grand\s*total|amount\s*due|cash)\b", s, flags=re.I) or re.search(r"\d+\.\d{2}", s):
        nums = re.findall(r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})|\d+\.\d{2}", s)
        if not nums:
            return None
        to_float = lambda x: float(re.sub(r"[,\s]", "", x))
        return max(map(to_float, nums))
    return None


# ------------------------------------------------------------------
# CHOOSE FIELD FOR EACH LINE
# ------------------------------------------------------------------
def choose_field_for_line(line: str, gt: EntityGT, line_idx: int = 0, total_lines: int = 0) -> Optional[str]:
    s = normalize_spaces(line)

    # Skip meta headers
    if re.search(r"(TAX\s*INVOICE|GST\s*ID|INVOICE\s*NO|DOCUMENT\s*NO)", s, flags=re.I):
        return None

    # DATE
    if is_date_like(s):
        return "DATE"

    # TOTAL
    amt = is_total_amount(s)
    if amt is not None and re.search(r"(total|grand\s*total|amount\s*due)", s, flags=re.I):
        return "TOTAL"

    # Fuzzy company/address matching
    c = score_company(s, gt.company)
    a = score_address(s, gt.address)

    # Lower thresholds for real-world OCR noise
    if c >= 55 and c >= a:
        return "COMPANY"
    if a >= 55 and a > c:
        return "ADDRESS"

    # Fallback positional priors
    pos = line_idx / max(total_lines, 1)
    if pos < 0.2:
        return "COMPANY"
    if 0.2 <= pos < 0.8:
        return "ADDRESS"
    return None


# ------------------------------------------------------------------
# MAIN MAPPING FUNCTION
# ------------------------------------------------------------------
def assign_lines_to_fields(lines, gt):
    """
    Robust fuzzy + positional mapping for COMPANY, DATE, ADDRESS, TOTAL.
    Returns {id(line): field_name}.
    """
    mapping = {}
    if not lines:
        return mapping

    ro_lines = sorted(lines, key=lambda l: l.aabb[1])  # top → bottom

    for idx, li in enumerate(ro_lines):
        s = normalize_spaces(li.text)
        if not s:
            continue

        # --- DATE ---
        if is_date_like(s) or re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", s):
            mapping[id(li)] = "DATE"
            continue

        # --- TOTAL ---
        if re.search(r"(total|grand\s*total|rounded\s*total|amount\s*due)", s, re.I) or re.search(r"RM\s*\d+\.\d{2}", s):
            mapping[id(li)] = "TOTAL"
            continue

        # --- COMPANY ---
        c_score = fuzz.partial_ratio(s.lower(), gt.company.lower()) if gt.company else 0
        if c_score >= 55:
            mapping[id(li)] = "COMPANY"
            continue

        # --- ADDRESS ---
        a_score = fuzz.token_set_ratio(s.lower(), gt.address.lower()) if gt.address else 0
        if a_score >= 50:
            mapping[id(li)] = "ADDRESS"
            continue

        # --- Fallback by position ---
        pos = idx / max(len(ro_lines), 1)
        if pos < 0.25:
            mapping[id(li)] = "COMPANY"
        elif pos < 0.75:
            mapping[id(li)] = "ADDRESS"
        else:
            mapping[id(li)] = "TOTAL"

    # mapped = sum(1 for v in mapping.values() if v)
    # missed = len(lines) - mapped
    # print(f"🧾 Debug summary: {mapped} mapped, {missed} missed.")
    return mapping


# ------------------------------------------------------------------
# LABEL MAPPING UTILITIES
# ------------------------------------------------------------------
def label_mappings():
    id2label = {i: l for i, l in enumerate(LABELS)}
    label2id = {l: i for i, l in id2label.items()}
    return label2id, id2label
