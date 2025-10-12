from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BoxLine:
    quad: Tuple[int,int,int,int,int,int,int,int]  # x1,y1,x2,y2,x3,y3,x4,y4
    text: str

    @property
    def aabb(self) -> Tuple[int,int,int,int]:
        xs = self.quad[0::2]; ys = self.quad[1::2]
        return min(xs), min(ys), max(xs), max(ys)

def _read_text_forgiving(path: str) -> str:
    """
    Try common encodings; last resort: binary decode with 'ignore'.
    """
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # Fallback: best-effort
    with open(path, "rb") as f:
        return f.read().decode("latin-1", errors="ignore")

def parse_box_file(path: str) -> List[BoxLine]:
    lines: List[BoxLine] = []
    raw_text = _read_text_forgiving(path)
    for raw in raw_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(",")
        # Expect at least 9 tokens: 8 coords + text (which may contain commas)
        if len(parts) < 9:
            continue
        try:
            coords = tuple(int(x) for x in parts[:8])
        except ValueError:
            # If any coord isn't an int (corrupted line), skip it
            continue

        text = ",".join(parts[8:]).strip()

        # 🧹 Skip empty or very short OCR text (e.g. ".", "-")
        if not text.strip() or len(text.strip()) < 2:
            continue

        lines.append(BoxLine(coords, text))
    return lines

def sort_reading_order(lines: List[BoxLine]) -> List[BoxLine]:
    def key(l: BoxLine):
        xmin, ymin, xmax, ymax = l.aabb
        return ((ymin + ymax) / 2.0, xmin)
    return sorted(lines, key=key)

def scale_boxes_to_0_1000(lines: List[BoxLine], img_w: int, img_h: int) -> List[Tuple[int,int,int,int,str]]:
    out = []
    for l in lines:
        xmin, ymin, xmax, ymax = l.aabb
        xmin = max(0, min(xmin, img_w-1)); xmax = max(0, min(xmax, img_w-1))
        ymin = max(0, min(ymin, img_h-1)); ymax = max(0, min(ymax, img_h-1))
        sxmin = int(xmin / img_w * 1000)
        sxmax = int(xmax / img_w * 1000)
        symin = int(ymin / img_h * 1000)
        symax = int(ymax / img_h * 1000)
        out.append((sxmin, symin, sxmax, symax, l.text))
    return out
