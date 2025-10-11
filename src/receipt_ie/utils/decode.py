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
    Strategy:
      - Merge consecutive B/I- tags into coherent spans
      - Prefer early spans for company/address, last span for total
      - Clean spacing/punctuation
    """
    if id2label is None:
        id2label = ID2LABEL

    spans: Dict[str, List[List[str]]] = {}
    cur_field: Optional[str] = None
    buf: List[str] = []

    def _flush(field, buf, spans):
        if field and buf:
            spans.setdefault(field, []).append(buf[:])
        buf.clear()

    # --- Build spans ---
    for w, li in zip(words, label_ids):
        if w == "[PAD]":
            continue  # ignore padding tokens

        lab = id2label.get(int(li), "O")
        if lab == "O":
            _flush(cur_field, buf, spans)
            cur_field = None
            continue

        prefix, _, tag = lab.partition("-")
        if prefix == "B":
            _flush(cur_field, buf, spans)
            cur_field = tag
            buf.append(w)
        elif prefix == "I":
            if cur_field == tag:
                buf.append(w)
            else:
                _flush(cur_field, buf, spans)
                cur_field = tag
                buf.append(w)
        else:
            _flush(cur_field, buf, spans)
            cur_field = None

    _flush(cur_field, buf, spans)

    # --- Pick best span per field ---
    out: Dict[str, str] = {}
    for tag in ("COMPANY", "DATE", "ADDRESS", "TOTAL"):
        cands = spans.get(tag, [])
        if not cands:
            out[tag.lower()] = ""
            continue

        # Prefer early spans for COMPANY/ADDRESS, last for TOTAL
        if tag in ("COMPANY", "ADDRESS"):
            best = cands[0]
        elif tag == "TOTAL":
            best = cands[-1]
        else:  # DATE
            best = max(cands, key=lambda s: len(s))

        # 🌟 NEW: limit company span length to prevent over-spanning
        if tag == "COMPANY" and len(best) > 10:
            best = best[:10]

        clean = " ".join(best).strip(" -:;,._")
        out[tag.lower()] = clean

    return out


