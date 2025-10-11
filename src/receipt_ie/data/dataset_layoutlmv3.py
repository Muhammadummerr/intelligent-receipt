# src/receipt_ie/data/dataset_layoutlmv3.py

import json, os
from typing import Dict, List, Tuple, Optional
from PIL import Image
from transformers import LayoutLMv3Processor
from .boxes import parse_box_file, sort_reading_order
from .align import assign_lines_to_fields, LABEL2ID, EntityGT, load_entities
from ..utils.text import split_tokens

FIELD_TO_BI = {
    "COMPANY": ("B-COMPANY","I-COMPANY"),
    "DATE": ("B-DATE","I-DATE"),
    "ADDRESS": ("B-ADDRESS","I-ADDRESS"),
    "TOTAL": ("B-TOTAL","I-TOTAL"),
}

# Accept common variants / cases
IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
BOX_EXTS = [".txt", ".TXT"]
ENT_EXTS = [".json", ".JSON", ".txt", ".TXT"]

def _find_with_ext(dirpath: str, stem: str, exts: List[str]) -> Optional[str]:
    for e in exts:
        p = os.path.join(dirpath, stem + e)
        if os.path.isfile(p):
            return p
    return None

class ReceiptLayoutLMv3Dataset:
    def __init__(self, img_dir: str, box_dir: str, ent_dir: str,
                 processor: LayoutLMv3Processor, max_seq_len: int = 1024,
                 stems_subset: Optional[List[str]] = None):
        self.img_dir = img_dir
        self.box_dir = box_dir
        self.ent_dir = ent_dir
        self.processor = processor
        self.max_seq_len = max_seq_len

        # Discover stems by actual files present (image + box + entities in ANY allowed ext)
        discovered = set()
        for name in os.listdir(img_dir):
            stem, ext = os.path.splitext(name)
            if ext not in IMG_EXTS:
                continue
            if _find_with_ext(box_dir, stem, BOX_EXTS) and _find_with_ext(ent_dir, stem, ENT_EXTS):
                discovered.add(stem)

        all_stems = sorted(discovered)
        if stems_subset:
            keep = set(stems_subset)
            self.stems = [s for s in all_stems if s in keep]
        else:
            self.stems = all_stems

    def __len__(self):
        return len(self.stems)

    def _read_item(self, idx: int):
        stem = self.stems[idx]

        img_path = _find_with_ext(self.img_dir, stem, IMG_EXTS)
        box_path = _find_with_ext(self.box_dir, stem, BOX_EXTS)
        ent_path = _find_with_ext(self.ent_dir, stem, ENT_EXTS)
        assert img_path and box_path and ent_path, f"Missing files for {stem}"

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        lines = sort_reading_order(parse_box_file(box_path))
        # entities may be in .json or .txt (but JSON content) — load with our helper
        ent = load_entities(ent_path)  # returns EntityGT

        return stem, image, W, H, lines, ent

    def __getitem__(self, idx: int):
        stem, image, W, H, lines, ent = self._read_item(idx)

        # 1) Scale line bboxes to [0,1000] and clamp safely
        scaled = []
        for li in lines:
            xmin, ymin, xmax, ymax = li.aabb
            sxmin = max(0, min(int(xmin / W * 1000), 1000))
            sxmax = max(0, min(int(xmax / W * 1000), 1000))
            symin = max(0, min(int(ymin / H * 1000), 1000))
            symax = max(0, min(int(ymax / H * 1000), 1000))
            scaled.append((sxmin, symin, sxmax, symax, li.text))

        # 2) Assign each line to a field (fixed id-based mapping)
        mapping = assign_lines_to_fields(lines, ent)

        # 3) Flatten into token-level lists
        words, boxes, labels = [], [], []
        for i, li in enumerate(lines):
            sxmin, symin, sxmax, symax, text = scaled[i]
            tokens = self.processor.tokenizer.tokenize(text)  # <- safer split
            if not tokens:
                continue

            field = mapping.get(id(li))  # <- fixed id-based mapping
            if field in FIELD_TO_BI:
                b_label, i_label = FIELD_TO_BI[field]
            else:
                b_label, i_label = "O", "O"

            for j, tok in enumerate(tokens):
                words.append(tok)
                boxes.append([sxmin, symin, sxmax, symax])
                if field in FIELD_TO_BI:
                    labels.append(LABEL2ID[b_label if j == 0 else i_label])
                else:
                    labels.append(LABEL2ID["O"])

        # 4) Truncate or pad safely
        if len(words) > self.max_seq_len:
            words = words[:self.max_seq_len]
            boxes = boxes[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(words)
            boxes += [[0, 0, 0, 0]] * pad_len
            words += ["[PAD]"] * pad_len
            labels += [-100] * pad_len  # ignore padding in loss

        # 5) Encode with processor
        encoded = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["id"] = stem
        return item

def label_mappings():
    return LABEL2ID, {v:k for k,v in LABEL2ID.items()}
