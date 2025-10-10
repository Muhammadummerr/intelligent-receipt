# src/receipt_ie/utils/decode.py
from typing import List, Dict, Optional

# Keep this consistent with training (align.LABELS / LABEL2ID)
LABELS = [
    "O",
    "B-COMPANY","I-COMPANY",
    "B-DATE","I-DATE",
    "B-ADDRESS","I-ADDRESS",
    "B-TOTAL","I-TOTAL",
]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

def _flush_span(field: Optional[str], buf: List[str], spans: Dict[str, List[List[str]]]):
    if field and buf:
        spans.setdefault(field, []).append(buf[:])
    buf.clear()

def group_bio(words: List[str], label_ids: List[int], id2label: Dict[int, str] = None) -> Dict[str, str]:
    """
    Convert word-level BIO ids into field strings.
    Returns dict with keys: company, date, address, total (lowercase).
    Strategy: collect all spans, then pick the longest span per field.
    """
    if id2label is None:
        id2label = ID2LABEL

    spans: Dict[str, List[List[str]]] = {}
    cur_field: Optional[str] = None
    buf: List[str] = []

    for w, li in zip(words, label_ids):
        lab = id2label.get(int(li), "O")
        if lab == "O":
            _flush_span(cur_field, buf, spans)
            cur_field = None
            continue

        prefix, _, tag = lab.partition("-")  # "B", "I" / "COMPANY" etc.
        if prefix == "B":
            _flush_span(cur_field, buf, spans)
            cur_field = tag
            buf.append(w)
        elif prefix == "I":
            # continue same field if matches; else start new (defensive)
            if cur_field == tag:
                buf.append(w)
            else:
                _flush_span(cur_field, buf, spans)
                cur_field = tag
                buf.append(w)
        else:
            # unknown -> flush
            _flush_span(cur_field, buf, spans)
            cur_field = None

    _flush_span(cur_field, buf, spans)

    # pick longest span per field
    out: Dict[str, str] = {}
    for tag in ("COMPANY", "DATE", "ADDRESS", "TOTAL"):
        cands = spans.get(tag, [])
        if not cands:
            out[tag.lower()] = ""
        else:
            best = max(cands, key=lambda s: len(s))
            out[tag.lower()] = " ".join(best)
    return out
