# src/receipt_ie/utils/postproc.py
import re
from typing import List, Dict

# must match your LABELS in align.py
ID2LABEL = {
    0:"O",
    1:"B-COMPANY", 2:"I-COMPANY",
    3:"B-DATE",    4:"I-DATE",
    5:"B-ADDRESS", 6:"I-ADDRESS",
    7:"B-TOTAL",   8:"I-TOTAL",
}

FIELD_FROM_LABEL = {
    "B-COMPANY":"company", "I-COMPANY":"company",
    "B-DATE":"date",       "I-DATE":"date",
    "B-ADDRESS":"address", "I-ADDRESS":"address",
    "B-TOTAL":"total",     "I-TOTAL":"total",
}

def group_bio(words: List[str], pred_ids: List[int]) -> Dict[str, str]:
    """
    Convert BIO token labels to 4 fields by concatenating contiguous spans.
    """
    fields = {"company":"", "date":"", "address":"", "total":""}
    cur_field = None
    buff: List[str] = []

    def flush():
        nonlocal buff, cur_field
        if cur_field and buff:
            s = " ".join(buff).strip()
            if s:
                if fields[cur_field]:
                    fields[cur_field] += " " + s
                else:
                    fields[cur_field] = s
        buff = []
        cur_field = None

    for w, pid in zip(words, pred_ids):
        tag = ID2LABEL.get(int(pid), "O")
        if tag == "O":
            flush()
            continue
        field = FIELD_FROM_LABEL[tag]
        if tag.startswith("B-"):
            flush()
            cur_field = field
            buff = [w]
        else:  # I-
            if cur_field == field:
                buff.append(w)
            else:
                # I- without B-; start new
                flush()
                cur_field = field
                buff = [w]
    flush()
    return fields

# --- Normalizers for eval/output consistency ---

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def norm_date(s: str) -> str:
    s = norm_spaces(s)
    # common receipts: DD/MM/YYYY, DD/MM/YY, YYYY-MM-DD, etc.
    # standardize to DD/MM/YYYY when possible
    # try DD/MM/YYYY or D/M/YYYY
    m = re.match(r"^\s*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\s*$", s)
    if m:
        d, mth, y = m.groups()
        if len(y) == 2:
            y = "20" + y
        return f"{int(d):02d}/{int(mth):02d}/{int(y):04d}"
    # try YYYY-MM-DD
    m = re.match(r"^\s*(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\s*$", s)
    if m:
        y, mth, d = m.groups()
        return f"{int(d):02d}/{int(mth):02d}/{int(y):04d}"
    return s

def norm_total(s: str) -> str:
    # pick largest-looking money amount with 2 decimals
    s = s or ""
    nums = re.findall(r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})|\d+\.\d{2}", s)
    if not nums:
        return norm_spaces(s)
    def to_float(x): return float(re.sub(r"[,\s]", "", x))
    amt = max(nums, key=to_float)
    return f"{to_float(amt):.2f}"
