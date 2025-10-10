# src/receipt_ie/data/dataset_infer.py
import torch
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
        lines = sort_reading_order(parse_box_file(box_path))

        # Build word-level inputs like training
        # Build word-level inputs like training
        words, boxes = [], []
        for li in lines:
            xmin, ymin, xmax, ymax = li.aabb

            # 1) clamp to image bounds
            xmin = max(0, min(xmin, W - 1))
            xmax = max(0, min(xmax, W - 1))
            ymin = max(0, min(ymin, H - 1))
            ymax = max(0, min(ymax, H - 1))

            # 2) fix inverted boxes if needed
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            if ymax < ymin:
                ymin, ymax = ymax, ymin

            # 3) ensure non-zero area in pixel space (avoid degenerate boxes)
            if xmax == xmin:
                xmax = min(W - 1, xmin + 1)
            if ymax == ymin:
                ymax = min(H - 1, ymin + 1)

            # 4) scale to 0..1000 (LayoutLM space) and clamp
            sxmin = int(xmin / W * 1000)
            sxmax = int(xmax / W * 1000)
            symin = int(ymin / H * 1000)
            symax = int(ymax / H * 1000)

            # final safety clamp (keep within [0,1000])
            sxmin = max(0, min(sxmin, 1000))
            sxmax = max(0, min(sxmax, 1000))
            symin = max(0, min(symin, 1000))
            symax = max(0, min(symax, 1000))

            toks = split_tokens(li.text)
            for _ in toks:
                words.append(_)
                boxes.append([sxmin, symin, sxmax, symax])


        # Truncate for safety
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

        # keep word_ids aligned to tokens (pad -> -1), version-safe
        enc0 = enc.encodings[0]  # EncodingFast
        wids_attr = getattr(enc0, "word_ids", None)
        if callable(wids_attr):
            wids = wids_attr()                      # older API: method
        else:
            wids = list(wids_attr) if wids_attr is not None else None  # newer API: list

        if wids is None:
            # fallback: no mapping available; align everything to -1
            wids = [None] * item["input_ids"].shape[0]

        item["word_ids"] = torch.tensor([(-1 if w is None else w) for w in wids], dtype=torch.long)
        item["orig_words"] = words
        item["id"] = stem
        return item


