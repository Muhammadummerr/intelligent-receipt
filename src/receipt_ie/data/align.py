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
def assign_lines_to_fields(lines: List[BoxLine], gt: EntityGT) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    ro_lines = sort_reading_order(lines)

    total_candidates = []
    for li in ro_lines:
        s = normalize_spaces(li.text)
        if is_date_like(s):
            mapping[id(li)] = "DATE"
            continue

        amt = is_total_amount(s)
        if amt is not None and re.search(r"(total|grand\s*total|amount\s*due|cash)", s, flags=re.I):
            total_candidates.append((amt, li, s))

    # Pick strongest total line
    if total_candidates:
        best_li = max(total_candidates, key=lambda x: (x[0], ro_lines.index(x[1])))[1]
        mapping[id(best_li)] = "TOTAL"

    # COMPANY / ADDRESS heuristic passes
    cutoff_triggered = False
    missed = []  # For debugging
    for idx, li in enumerate(ro_lines):
        if id(li) in mapping:
            continue
        s = normalize_spaces(li.text)

        if re.search(r"(TOTAL|AMOUNT|INVOICE|GST|CASH|ROUNDING|CHANGE)", s, flags=re.I):
            cutoff_triggered = True

        if idx < 5:
            mapping[id(li)] = "COMPANY"
            continue

        if cutoff_triggered and "ADDRESS" in mapping.values():
            continue

        if re.search(r"\d{4,5}|JALAN|TAMAN|SELANGOR|KUALA\s*LUMPUR|MALAYSIA", s, flags=re.I):
            mapping[id(li)] = "ADDRESS"
            continue

        f = choose_field_for_line(s, gt, idx, len(ro_lines))
        if f:
            mapping[id(li)] = f
        else:
            missed.append(s)

    # Debug summary for first few samples
    if len(ro_lines) < 10:  # small receipts
        print(f"\n🧾 Debug summary: {len(mapping)} mapped lines, {len(missed)} missed.")
        if missed:
            print("⚠️ Missed text lines:")
            for t in missed[:5]:
                print("  ", t)

    return mapping


# ------------------------------------------------------------------
# LABEL MAPPING UTILITIES
# ------------------------------------------------------------------
def label_mappings():
    id2label = {i: l for i, l in enumerate(LABELS)}
    label2id = {l: i for i, l in id2label.items()}
    return label2id, id2label
