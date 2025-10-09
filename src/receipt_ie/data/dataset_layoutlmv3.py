import json, os
from typing import Dict, List, Tuple
from PIL import Image
from transformers import LayoutLMv3Processor
from .boxes import parse_box_file, sort_reading_order, scale_boxes_to_0_1000
from .align import assign_lines_to_fields, LABEL2ID
from ..utils.text import split_tokens, normalize_spaces

FIELD_TO_BI = {
    "COMPANY": ("B-COMPANY","I-COMPANY"),
    "DATE": ("B-DATE","I-DATE"),
    "ADDRESS": ("B-ADDRESS","I-ADDRESS"),
    "TOTAL": ("B-TOTAL","I-TOTAL"),
}

class ReceiptLayoutLMv3Dataset:
    def __init__(self, img_dir: str, box_dir: str, ent_dir: str,
                 processor: LayoutLMv3Processor, max_seq_len: int = 1024,
                 stems_subset: List[str] = None):
        self.img_dir = img_dir
        self.box_dir = box_dir
        self.ent_dir = ent_dir
        self.processor = processor
        self.max_seq_len = max_seq_len

        all_stems = []
        for name in os.listdir(img_dir):
            stem, ext = os.path.splitext(name)
            if not os.path.isfile(os.path.join(box_dir, stem + ".txt")):
                continue
            if not os.path.isfile(os.path.join(ent_dir, stem + ".json")):
                continue
            all_stems.append(stem)
        all_stems = sorted(all_stems)
        if stems_subset:
            keep = set(stems_subset)
            self.stems = [s for s in all_stems if s in keep]
        else:
            self.stems = all_stems


    def __len__(self):
        return len(self.stems)

    def _read_item(self, idx: int):
        stem = self.stems[idx]
        img_path = os.path.join(self.img_dir, stem + ".jpg")
        if not os.path.exists(img_path):
            # fallback to png
            img_path = os.path.join(self.img_dir, stem + ".png")
        box_path = os.path.join(self.box_dir, stem + ".txt")
        ent_path = os.path.join(self.ent_dir, stem + ".json")

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        lines = sort_reading_order(parse_box_file(box_path))
        with open(ent_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        return stem, image, W, H, lines, gt

    def __getitem__(self, idx: int):
        stem, image, W, H, lines, gt = self._read_item(idx)

        # 1) scale bboxes to 0..1000 space
        scaled = []
        for li in lines:
            xmin, ymin, xmax, ymax = li.aabb
            sxmin = int(xmin / W * 1000); sxmax = int(xmax / W * 1000)
            symin = int(ymin / H * 1000); symax = int(ymax / H * 1000)
            scaled.append((sxmin, symin, sxmax, symax, li.text))

        # 2) assign each line to a field (or None)
        from .align import EntityGT, load_entities, assign_lines_to_fields, LABEL2ID
        ent = load_entities(os.path.join(self.ent_dir, stem + ".json"))
        mapping = assign_lines_to_fields(lines, ent)

        # 3) flatten into token-level lists
        words, boxes, labels = [], [], []
        for i, li in enumerate(lines):
            sxmin, symin, sxmax, symax, text = scaled[i]
            tokens = split_tokens(text)
            if not tokens:
                continue
            field = mapping.get(i)  # may be None
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

        # 4) truncate to max_seq_len (keep early tokens)
        if len(words) > self.max_seq_len:
            words = words[:self.max_seq_len]
            boxes = boxes[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        encoded = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        # flatten batch dim
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["id"] = stem
        return item

def label_mappings():
    return LABEL2ID, {v:k for k,v in LABEL2ID.items()}
