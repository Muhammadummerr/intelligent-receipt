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
    # normalize separators and zero-pad day/month, keep year as-is
    m = re.match(r"^\s*(\d{1,4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,4})\s*$", s)
    if m:
        a, b, c = m.groups()
        # try common orders; prefer dd/mm/yy(yy) if plausible
        # If first is 1-2 digits and last is 2-4 digits, assume D/M/Y
        if len(a) <= 2 and len(c) in (2,3,4):
            return f"{int(a):02d}/{int(b):02d}/{c}"
        # If first is 4 digits, assume Y-M-D -> D/M/Y
        if len(a) == 4 and len(c) <= 2:
            return f"{int(c):02d}/{int(b):02d}/{a}"
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
