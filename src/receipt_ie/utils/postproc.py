# src/receipt_ie/utils/postproc.py
import re

WS = re.compile(r"\s+")
def norm_spaces(s: str) -> str:
    return WS.sub(" ", (s or "").strip())

# ----------------------
# DATE
# ----------------------
# Accepts: 20/04/2018, 20-04-18, 2018/04/20, "14/02/2018 5:37:44PM", "DATE: 01/02/2018",
#          "29 JUN 18", "APR 20, 2018"
_DATE_TOK = r"(?:\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})"
DATE_LIKE = re.compile(_DATE_TOK, flags=re.IGNORECASE)

def soft_date_norm(s: str) -> str:
    s = norm_spaces(s)
    m = re.match(r"^\s*(\d{1,4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,4})\s*$", s)
    if m:
        a,b,c = m.groups()
        # If D/M/Y (a up to 2 digits; keep year as-is: 2 or 4 digits)
        if len(a) <= 2 and 2 <= len(c) <= 4:
            return f"{int(a):02d}/{int(b):02d}/{c}"
        # If Y-M-D -> D/M/Y
        if len(a) == 4 and len(c) <= 2:
            return f"{int(c):02d}/{int(b):02d}/{a}"
    # Month name formats -> keep as is (no risky conversion)
    return s

def extract_best_date(s: str) -> str:
    """Pick a single date-like token from a noisy string and normalize gently."""
    s_norm = norm_spaces(s)
    hits = DATE_LIKE.findall(s_norm)
    if not hits:
        return ""
    # simple choice: prefer the first date token (top-left bias)
    # special case: if there are duplicates, keep unique
    first = hits[0]
    return soft_date_norm(first)

# ----------------------
# TOTAL
# ----------------------
NUM2 = re.compile(r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})|\d+\.\d{2}")
CURRENCY = re.compile(r"^(?:RM|MYR|USD|\$)\s*", flags=re.IGNORECASE)

def _to_float(x: str) -> float:
    return float(re.sub(r"[,\s]", "", x))

TOTAL_KEY = re.compile(r"\b(total|grand\s*total|amount\s*due|cash|balance\s*due|total\s*sales)\b", re.IGNORECASE)
NEGATIVE_HINT = re.compile(r"\b(gst|tax|vat|discount|rounding|change)\b", re.IGNORECASE)

def soft_total_norm(s: str) -> str:
    s = norm_spaces(s)
    s = CURRENCY.sub("", s)
    nums = NUM2.findall(s)
    if not nums: 
        return ""
    best = max(nums, key=_to_float)  # largest 2-dec number
    return f"{_to_float(best):.2f}"

def pick_total_from_lines(lines: list) -> str:
    """
    Given a list of text lines (e.g., around predicted TOTAL span),
    prefer lines with TOTAL-like keywords and no negative hints; among numbers pick largest < 100000.
    """
    cand = []
    for t in lines:
        t0 = norm_spaces(t)
        score = 0
        if TOTAL_KEY.search(t0):
            score += 10
        if NEGATIVE_HINT.search(t0):
            score -= 5
        nums = NUM2.findall(t0)
        if not nums:
            continue
        vals = [_to_float(n) for n in nums]
        top = max(vals)
        if top < 100000:  # sanity
            score += min(int(top // 10), 10)  # mild bias to larger totals
            cand.append((score, top, t0))
    if not cand:
        return ""
    cand.sort(reverse=True)  # highest score first
    return f"{cand[0][1]:.2f}"

# ----------------------
# COMPANY
# ----------------------
LEGAL_JUNK = re.compile(
    r"\s*\(.*?\)\s*$|"
    r"\b(SDN\s*BHD|SDN\.?\s*BHD\.?|BHD|LTD|CO\.?|PTE|S\/B|S\.?\/?B\.?)\b.*$|"
    r"\b(ITEM\s*TOTAL|TOTAL\s*ITEM|CHANGE|CASHIER|DOC\s*NO\.?)\b.*$",
    re.IGNORECASE
)

def clean_company(s: str) -> str:
    s = norm_spaces(s)
    s = LEGAL_JUNK.sub("", s).strip(" -:|")
    return s
