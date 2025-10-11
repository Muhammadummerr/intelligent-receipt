import json, os, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from rapidfuzz import fuzz
from .boxes import BoxLine, sort_reading_order
from ..utils.text import normalize_spaces, strip_punct_lower, is_date_like

FIELDS = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]
LABELS = ["O",
          "B-COMPANY","I-COMPANY",
          "B-DATE","I-DATE",
          "B-ADDRESS","I-ADDRESS",
          "B-TOTAL","I-TOTAL"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}

@dataclass
class EntityGT:
    company: str
    date: str
    address: str
    total: str

def load_entities(path: str) -> EntityGT:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    # helper: get first non-empty value among candidate keys
    def pick(keys, default=""):
        for k in keys:
            if k in j and j[k] is not None:
                val = str(j[k]).strip()
                if val != "":
                    return val
        return default

    company = pick(["company", "vendor", "merchant", "store", "company_name"])
    date    = pick(["date", "invoice_date", "receipt_date"])
    address = pick(["address", "addr", "location", "address_1", "address1"])
    total   = pick(["total", "amount", "total_amount", "grand_total", "total_sales", "total_sale"])

    return EntityGT(
        company=normalize_spaces(company),
        date=normalize_spaces(date),
        address=normalize_spaces(address),
        total=normalize_spaces(total),
    )


# --- Heuristics to score which line belongs to which field ---

def score_company(line: str, gt_company: str) -> float:
    return fuzz.partial_ratio(line, gt_company)

def score_address(line: str, gt_address: str) -> float:
    return fuzz.token_set_ratio(line, gt_address)

def is_total_amount(line: str) -> Optional[float]:
    # Return numeric amount if line looks like a total candidate, else None
    if re.search(r"\b(total|grand\s*total|amount\s*due|cash)\b", line, flags=re.I) or re.search(r"\b\d+\.\d{2}\b", line):
        # pick the largest number with 2 decimals
        nums = re.findall(r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})|\d+\.\d{2}", line)
        if not nums:
            return None
        def to_float(x):
            return float(re.sub(r"[,\s]", "", x))
        return max(map(to_float, nums))
    return None

def choose_field_for_line(line: str, gt: EntityGT, line_idx: int = 0, total_lines: int = 0) -> Optional[str]:
    s = normalize_spaces(line)

    # DATE
    if is_date_like(s):
        return "DATE"

    # TOTAL
    amt = is_total_amount(s)
    if amt is not None:
        if re.search(r"\b(total|grand\s*total|amount\s*due)\b", s, flags=re.I):
            return "TOTAL"

    # COMPANY vs ADDRESS: position-aware + fuzzy fallback
    c = score_company(s, gt.company)
    a = score_address(s, gt.address)

    # ↓ allow weaker fuzzy matches (was 45–60) and use line position hints ↓
    if max(c, a) < 40:
        pos = line_idx / max(total_lines, 1)
        # Top 25 % → COMPANY
        if pos < 0.25:
            return "COMPANY"
        # Middle 25–80 % → ADDRESS (most addresses live here)
        if 0.25 <= pos <= 0.8:
            return "ADDRESS"
        # Bottom 20 % → likely totals
        return None

    return "COMPANY" if c >= a else "ADDRESS"


def assign_lines_to_fields(lines: List[BoxLine], gt: EntityGT) -> Dict[int, str]:
    """
    Return mapping {id(line) -> field_name or None}.
    Uses id(li) instead of index to avoid mismatch after sorting.
    Adds light positional priors for COMPANY and ADDRESS.
    """
    mapping: Dict[int, str] = {}
    ro_lines = sort_reading_order(lines)

    # --- Pass 1 : DATE / TOTAL ---
    for li in ro_lines:
        s = normalize_spaces(li.text)
        if is_date_like(s):
            mapping[id(li)] = "DATE"
            continue
        if is_total_amount(s) is not None and re.search(
            r"\b(total|grand\s*total|amount\s*due|cash)\b", s, flags=re.I
        ):
            mapping[id(li)] = "TOTAL"

    # --- Pass 2 : COMPANY / ADDRESS (with bias) ---
    for idx, li in enumerate(ro_lines):
        if id(li) in mapping:
            continue

        # Early lines (top 5) strongly likely COMPANY
        if idx < 5:
            mapping[id(li)] = "COMPANY"
            continue

        f = choose_field_for_line(li.text, gt, idx, len(ro_lines))

        # 🌟 NEW: address fallback heuristic (detect postal/address patterns)
        if f == "ADDRESS":
            s = normalize_spaces(li.text)
            if re.search(r"\d{4,5}|JALAN|TAMAN|SELANGOR|KUALA\s*LUMPUR|MALAYSIA", s, flags=re.I):
                mapping[id(li)] = "ADDRESS"
                continue

        # if a weak fuzzy match still exists, keep it
        if f:
            mapping[id(li)] = f

    return mapping


def label_mappings():
    id2label = {i: l for i, l in enumerate(LABELS)}
    label2id = {l: i for i, l in id2label.items()}
    return label2id, id2label

