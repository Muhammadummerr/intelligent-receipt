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

def choose_field_for_line(line: str, gt: EntityGT) -> Optional[str]:
    s = normalize_spaces(line)
    # DATE
    if is_date_like(s):
        return "DATE"
    # TOTAL
    amt = is_total_amount(s)
    if amt is not None:
        # weak prior: lines with TOTAL-like words get TOTAL
        if re.search(r"\b(total|grand\s*total|amount\s*due)\b", s, flags=re.I):
            return "TOTAL"
    # COMPANY vs ADDRESS: pick whichever has higher fuzzy score
    c = score_company(s, gt.company)
    a = score_address(s, gt.address)
    if max(c, a) < 60:
        return None
    return "COMPANY" if c >= a else "ADDRESS"

def assign_lines_to_fields(lines: List[BoxLine], gt: EntityGT) -> Dict[int, str]:
    """Return mapping line_index -> field or None."""
    mapping: Dict[int, str] = {}
    ro_lines = sort_reading_order(lines)
    # first pass: date / total confident
    for idx, li in enumerate(ro_lines):
        s = normalize_spaces(li.text)
        if is_date_like(s):
            mapping[idx] = "DATE"
            continue
        if is_total_amount(s) is not None and re.search(r"\b(total|grand\s*total|amount\s*due|cash)\b", s, flags=re.I):
            mapping[idx] = "TOTAL"
    # second pass: company/address with fuzzy scores
    for idx, li in enumerate(ro_lines):
        if idx in mapping: 
            continue
        f = choose_field_for_line(li.text, gt)
        if f:
            mapping[idx] = f
    return mapping
