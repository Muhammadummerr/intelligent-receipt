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
    s_norm = norm_spaces(s)
    hits = DATE_LIKE.findall(s_norm)
    if not hits:
        return ""
    # prefer tokens that look like day/month/year or include a month name + year
    def score(tok: str) -> int:
        t = tok.upper()
        sc = 0
        if re.search(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b", t):
            sc += 2
        if re.search(r"\d{4}$|\d{2}$", t):  # ends with year-ish
            sc += 1
        # penalize patterns like "MAR 18 12" (month + 2 numbers)
        if re.search(r"^[A-Z]{3,9}\s+\d{1,2}\s+\d{2}$", t):
            sc -= 1
        return sc
    hits.sort(key=score, reverse=True)
    return soft_date_norm(hits[0])


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

def pick_total_from_lines(lines):
    """
    Pick the best total line, preferring ones containing TOTAL/GRAND TOTAL,
    or bottom-most numeric lines.
    """
    best = ""
    best_val = 0.0
    for ln in lines:
        if not ln.strip():
            continue
        m = re.search(r"(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2}))", ln)
        if m:
            val = float(re.sub(r"[,\s]", "", m.group(1)))
            if re.search(r"\b(total|grand|amount|cash)\b", ln, flags=re.I):
                # strong keyword boost
                val += 0.001
            if val >= best_val:
                best_val = val
                best = m.group(1)
    return best


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
    
    # remove copy/duplicate/original markers often in header lines
    s = re.sub(r"\b(COPY|DUPLICATE|ORIGINAL)\b.*$", "", s, flags=re.I)
    
    # remove tax/legal junk (company suffixes etc.)
    s = re.sub(r"\b(SDN\s*BHD|SDN|BHD|LTD|LIMITED|ENTERPRISE|ENTERPRISES|PLT|CO\.?|M)\b.*", "", s, flags=re.I)
    
    # cut off at 'TEL', 'FAX', 'GST', 'TAX INVOICE', 'RECEIPT', etc.
    s = re.split(r"\b(TEL|FAX|GST|TAX|RECEIPT|INVOICE|DATE|TIME|NO\.)\b", s)[0]
    
    # keep only uppercase letters, numbers, & spaces
    s = re.sub(r"[^A-Z0-9\s\.\-&]", "", s, flags=re.I)
    
    return s.strip()




# ---- compatibility aliases----
def norm_date(s: str) -> str:
    return soft_date_norm(s)

def norm_total(s: str) -> str:
    return soft_total_norm(s)
