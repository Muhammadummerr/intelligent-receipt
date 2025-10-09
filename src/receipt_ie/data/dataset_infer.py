# src/receipt_ie/data/dataset_infer.py
import os
from typing import List, Optional
from PIL import Image
from transformers import LayoutLMv3Processor
from .boxes import parse_box_file, sort_reading_order
from ..utils.text import split_tokens

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
BOX_EXTS = [".txt", ".TXT"]

def _find_with_ext(dirpath: str, stem: str, exts: List[str]) -> Optional[str]:
    for e in exts:
        p = os.path.join(dirpath, stem + e)
        if os.path.isfile(p):
            return p
    return None

class ReceiptInferenceDataset:
    """
    Produces inputs required by LayoutLMv3 for inference (no labels).
    Returns dict with: input_ids, attention_mask, bbox, pixel_values, and id.
    """
    def __init__(self, img_dir: str, box_dir: str,
                 processor: LayoutLMv3Processor, max_seq_len: int = 512,
                 stems_subset: Optional[List[str]] = None):
        self.img_dir = img_dir
        self.box_dir = box_dir
        self.processor = processor
        self.max_seq_len = max_seq_len

        discovered = set()
        for name in os.listdir(img_dir):
            stem, ext = os.path.splitext(name)
            if ext not in IMG_EXTS:
                continue
            if _find_with_ext(box_dir, stem, BOX_EXTS):
                discovered.add(stem)

        all_stems = sorted(discovered)
        if stems_subset:
            keep = set(stems_subset)
            self.stems = [s for s in all_stems if s in keep]
        else:
            self.stems = all_stems

    def __len__(self): return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img_path = _find_with_ext(self.img_dir, stem, IMG_EXTS)
        box_path = _find_with_ext(self.box_dir, stem, BOX_EXTS)
        assert img_path and box_path, f"Missing files for {stem}"

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # read OCR lines and keep reading order
        lines = sort_reading_order(parse_box_file(box_path))

        # words + bbox per word (tokenization at 'word' level like training)
        words, boxes = [], []
        for li in lines:
            xmin, ymin, xmax, ymax = li.aabb
            sxmin = int(xmin / W * 1000); sxmax = int(xmax / W * 1000)
            symin = int(ymin / H * 1000); symax = int(ymax / H * 1000)
            toks = split_tokens(li.text)
            for _ in toks:
                words.append(_)
                boxes.append([sxmin, symin, sxmax, symax])

        # truncate (LayoutLMv3 needs equal lengths)
        if len(words) > self.max_seq_len:
            words = words[:self.max_seq_len]
            boxes = boxes[:self.max_seq_len]

        enc = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["id"] = stem
        return item
