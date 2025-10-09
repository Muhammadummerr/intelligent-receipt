from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class BoxLine:
    quad: Tuple[int,int,int,int,int,int,int,int]  # x1,y1, x2,y2, x3,y3, x4,y4
    text: str

    @property
    def aabb(self) -> Tuple[int,int,int,int]:
        xs = self.quad[0::2]; ys = self.quad[1::2]
        return min(xs), min(ys), max(xs), max(ys)

def parse_box_file(path: str) -> List[BoxLine]:
    lines: List[BoxLine] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(",")
            # last item is text (may contain commas already if not last 9 items)
            if len(parts) < 9:
                # malformed line, skip
                continue
            coords = list(map(int, parts[:8]))
            text = ",".join(parts[8:]).strip()
            lines.append(BoxLine(tuple(coords), text))
    return lines

def sort_reading_order(lines: List[BoxLine]) -> List[BoxLine]:
    # sort by vertical center, then left x
    def key(l: BoxLine):
        xmin, ymin, xmax, ymax = l.aabb
        return ( (ymin+ymax)/2.0, xmin )
    return sorted(lines, key=key)

def scale_boxes_to_0_1000(lines: List[BoxLine], img_w: int, img_h: int) -> List[Tuple[int,int,int,int,str]]:
    """Return list of (xmin,xmax,ymin,ymax,text) scaled to 0..1000 (LayoutLM space)."""
    out = []
    for l in lines:
        xmin, ymin, xmax, ymax = l.aabb
        # clamp
        xmin = max(0, min(xmin, img_w-1)); xmax = max(0, min(xmax, img_w-1))
        ymin = max(0, min(ymin, img_h-1)); ymax = max(0, min(ymax, img_h-1))
        # scale
        sxmin = int(xmin / img_w * 1000)
        sxmax = int(xmax / img_w * 1000)
        symin = int(ymin / img_h * 1000)
        symax = int(ymax / img_h * 1000)
        out.append((sxmin, symin, sxmax, symax, l.text))
    return out
