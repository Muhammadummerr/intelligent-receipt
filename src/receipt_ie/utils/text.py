import re
from typing import List

SPACE_RE = re.compile(r"\s+")

def normalize_spaces(s: str) -> str:
    return SPACE_RE.sub(" ", s.strip())

def strip_punct_lower(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s).lower()

def split_tokens(line: str) -> List[str]:
    # conservative tokenization: words and numbers, keep punctuation attached if needed
    return [t for t in re.split(r"(\s+)", line) if t.strip()]

def is_date_like(s: str) -> bool:
    return bool(re.search(r"\b\d{1,2}([/\-\.])\d{1,2}\1(\d{2}|\d{4})\b", s))

def normalize_date(s: str) -> str:
    # Return as DD/MM/YYYY when possible, else original
    m = re.search(r"\b(\d{1,2})([/\-\.])(\d{1,2})\2(\d{2}|\d{4})\b", s)
    if not m:
        return s
    d, sep, mth, y = m.group(1), m.group(2), m.group(3), m.group(4)
    d = d.zfill(2); mth = mth.zfill(2)
    if len(y) == 2:
        # pivot year 00-24 -> 2000-2024; else 1900s
        y2 = int(y)
        y = 2000 + y2 if y2 <= 24 else 1900 + y2
    return f"{d}/{mth}/{y}"
