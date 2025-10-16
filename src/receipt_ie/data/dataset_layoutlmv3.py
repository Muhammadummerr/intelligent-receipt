# # src/receipt_ie/data/dataset_layoutlmv3.py

# import json, os
# from typing import Dict, List, Tuple, Optional
# from PIL import Image
# from transformers import LayoutLMv3Processor
# from .boxes import parse_box_file, sort_reading_order
# # at the top of dataset_layoutlmv3.py
# from .align import assign_lines_to_fields, LABEL2ID, EntityGT, load_entities, label_mappings

# from ..utils.text import split_tokens

# FIELD_TO_BI = {
#     "COMPANY": ("B-COMPANY","I-COMPANY"),
#     "DATE": ("B-DATE","I-DATE"),
#     "ADDRESS": ("B-ADDRESS","I-ADDRESS"),
#     "TOTAL": ("B-TOTAL","I-TOTAL"),
# }

# # Accept common variants / cases
# IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
# BOX_EXTS = [".txt", ".TXT"]
# ENT_EXTS = [".json", ".JSON", ".txt", ".TXT"]

# def _find_with_ext(dirpath: str, stem: str, exts: List[str]) -> Optional[str]:
#     for e in exts:
#         p = os.path.join(dirpath, stem + e)
#         if os.path.isfile(p):
#             return p
#     return None

# class ReceiptLayoutLMv3Dataset:
#     def __init__(self, img_dir: str, box_dir: str, ent_dir: str,
#                  processor: LayoutLMv3Processor, max_seq_len: int = 1024,
#                  stems_subset: Optional[List[str]] = None):
#         self.img_dir = img_dir
#         self.box_dir = box_dir
#         self.ent_dir = ent_dir
#         self.processor = processor
#         self.max_seq_len = max_seq_len

#         # Discover stems by actual files present (image + box + entities in ANY allowed ext)
#         discovered = set()
#         for name in os.listdir(img_dir):
#             stem, ext = os.path.splitext(name)
#             if ext not in IMG_EXTS:
#                 continue
#             if _find_with_ext(box_dir, stem, BOX_EXTS) and _find_with_ext(ent_dir, stem, ENT_EXTS):
#                 discovered.add(stem)

#         all_stems = sorted(discovered)
#         if stems_subset:
#             keep = set(stems_subset)
#             self.stems = [s for s in all_stems if s in keep]
#         else:
#             self.stems = all_stems

#     def __len__(self):
#         return len(self.stems)

#     def _read_item(self, idx: int):
#         stem = self.stems[idx]

#         img_path = _find_with_ext(self.img_dir, stem, IMG_EXTS)
#         box_path = _find_with_ext(self.box_dir, stem, BOX_EXTS)
#         ent_path = _find_with_ext(self.ent_dir, stem, ENT_EXTS)
#         assert img_path and box_path and ent_path, f"Missing files for {stem}"

#         image = Image.open(img_path).convert("RGB")
#         W, H = image.size

#         lines = sort_reading_order(parse_box_file(box_path))
#         # entities may be in .json or .txt (but JSON content) — load with our helper
#         ent = load_entities(ent_path)  # returns EntityGT

#         return stem, image, W, H, lines, ent

#     def __getitem__(self, idx: int):
#         stem, image, W, H, lines, ent = self._read_item(idx)

#         # 1) Scale line bboxes to [0,1000] and clamp safely
#         scaled = []
#         for li in lines:
#             xmin, ymin, xmax, ymax = li.aabb
#             sxmin = max(0, min(int(xmin / W * 1000), 1000))
#             sxmax = max(0, min(int(xmax / W * 1000), 1000))
#             symin = max(0, min(int(ymin / H * 1000), 1000))
#             symax = max(0, min(int(ymax / H * 1000), 1000))
#             scaled.append((sxmin, symin, sxmax, symax, li.text))

#         # 2) Assign each line to a field (fixed id-based mapping)
#         mapping = assign_lines_to_fields(lines, ent)

#         # 3) Flatten into token-level lists
#         words, boxes, labels = [], [], []
#         for i, li in enumerate(lines):
#             sxmin, symin, sxmax, symax, text = scaled[i]
#             clean_text = " ".join(text.split())  # remove redundant spaces/newlines
#             tokens = self.processor.tokenizer.tokenize(clean_text)
#             if not tokens:
#                 continue

#             field = mapping.get(id(li))  # <- fixed id-based mapping
#             if field in FIELD_TO_BI:
#                 b_label, i_label = FIELD_TO_BI[field]
#             else:
#                 b_label, i_label = "O", "O"

#             for j, tok in enumerate(tokens):
#                 words.append(tok)
#                 boxes.append([sxmin, symin, sxmax, symax])
#                 if field in FIELD_TO_BI:
#                     labels.append(LABEL2ID[b_label if j == 0 else i_label])
#                 else:
#                     labels.append(LABEL2ID["O"])

#         # 4) Truncate or pad safely
#         if len(words) > self.max_seq_len:
#             words = words[:self.max_seq_len]
#             boxes = boxes[:self.max_seq_len]
#             labels = labels[:self.max_seq_len]
#         else:
#             pad_len = self.max_seq_len - len(words)
#             boxes += [[0, 0, 0, 0]] * pad_len
#             words += ["[PAD]"] * pad_len
#             labels += [-100] * pad_len  # ignore padding in loss

#         # 5) Encode with processor
#         encoded = self.processor(
#             image,
#             words,
#             boxes=boxes,
#             word_labels=labels,
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_seq_len,
#             return_tensors="pt",
#         )

#         item = {k: v.squeeze(0) for k, v in encoded.items()}
#         item["id"] = stem
#         return item


import os, json
from typing import Dict, List, Tuple, Optional
from PIL import Image
from transformers import LayoutLMv3Processor
from .boxes import parse_box_file, sort_reading_order
from .align import assign_lines_to_fields, LABEL2ID, EntityGT, load_entities
from ..utils.text import split_tokens
import easyocr
import torch

FIELD_TO_BI = {
    "COMPANY": ("B-COMPANY", "I-COMPANY"),
    "DATE": ("B-DATE", "I-DATE"),
    "ADDRESS": ("B-ADDRESS", "I-ADDRESS"),
    "TOTAL": ("B-TOTAL", "I-TOTAL"),
}

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
    def __init__(
        self,
        img_dir: str,
        box_dir: str,
        ent_dir: str,
        processor: LayoutLMv3Processor,
        max_seq_len: int = 1024,
        stems_subset: Optional[List[str]] = None,
        use_easyocr: bool = False,
        cache_dir: str = "./ocr_cache"
    ):
        """
        use_easyocr=True → dynamically extract OCR text & boxes via EasyOCR
        cache_dir → stores OCR results to avoid re-running EasyOCR every time
        """
        self.img_dir = img_dir
        self.box_dir = box_dir
        self.ent_dir = ent_dir
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.use_easyocr = use_easyocr
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        # discover all valid stems
        discovered = set()
        for name in os.listdir(img_dir):
            stem, ext = os.path.splitext(name)
            if ext not in IMG_EXTS:
                continue
            if use_easyocr or (_find_with_ext(box_dir, stem, BOX_EXTS) and _find_with_ext(ent_dir, stem, ENT_EXTS)):
                discovered.add(stem)

        self.stems = sorted(discovered)

        # initialize OCR reader if needed
        self.reader = None
        if use_easyocr:
            print("🧠 Initializing EasyOCR reader for dynamic OCR + caching...")
            self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def __len__(self):
        return len(self.stems)

    # --------------------- OCR Caching Helper --------------------- #
    def _get_cached_ocr(self, img_path: str, stem: str, W: int, H: int):
        """Return OCR (words, boxes) from cache or run EasyOCR if not cached."""
        cache_path = os.path.join(self.cache_dir, f"{stem}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if "words" in cached and "boxes" in cached:
                        return cached["words"], cached["boxes"]
            except Exception:
                print(f"⚠️ Cache corrupted for {stem}, regenerating...")

        print(f"🧩 Running EasyOCR for {stem}...")
        results = self.reader.readtext(img_path, detail=1, paragraph=False)

        words, boxes = [], []
        for (bbox, text, conf) in results:
            if not text.strip():
                continue
            ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = bbox
            xmin, ymin = min(x1, x2, x3, x4), min(y1, y2, y3, y4)
            xmax, ymax = max(x1, x2, x3, x4), max(y1, y2, y3, y4)
            boxes.append([
                int(xmin / W * 1000),
                int(ymin / H * 1000),
                int(xmax / W * 1000),
                int(ymax / H * 1000),
            ])
            words.append(text.strip())

        # save to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"words": words, "boxes": boxes}, f, ensure_ascii=False, indent=2)

        return words, boxes

    # -------------------------------------------------------------- #
    def _read_item(self, idx: int):
        stem = self.stems[idx]
        img_path = _find_with_ext(self.img_dir, stem, IMG_EXTS)
        ent_path = _find_with_ext(self.ent_dir, stem, ENT_EXTS)
        assert img_path and ent_path, f"Missing files for {stem}"

        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        ent = load_entities(ent_path)

        if self.use_easyocr:
            words, boxes = self._get_cached_ocr(img_path, stem, W, H)
            lines = [{"aabb": (b[0], b[1], b[2], b[3]), "text": w} for w, b in zip(words, boxes)]
        else:
            box_path = _find_with_ext(self.box_dir, stem, BOX_EXTS)
            from .boxes import BoxLine
            parsed = sort_reading_order(parse_box_file(box_path))
            lines = [{"aabb": li.aabb, "text": li.text} for li in parsed]

        return stem, image, W, H, lines, ent

    # -------------------------------------------------------------- #
    def __getitem__(self, idx: int):
        from .boxes import BoxLine
        import torch

        stem, image, W, H, lines, ent = self._read_item(idx)

        # --- Caching: skip recomputation if cached ---
        cache_path = os.path.join(self.cache_dir, f"{stem}.pt")
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path)
            except Exception:
                print(f"⚠️ Corrupted cache for {stem}, regenerating...")

        # --- Properly expand AABB to 8-coordinate quad (rectangle) ---
        box_lines = []
        for li in lines:
            xmin, ymin, xmax, ymax = map(int, li["aabb"])
            quad = (xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax)
            box_lines.append(BoxLine(quad, li["text"]))

        # --- Field mapping ---
        mapping = assign_lines_to_fields(box_lines, ent)

        words, boxes, labels = [], [], []

        for li in box_lines:
            xmin, ymin, xmax, ymax = li.aabb

            # 🧱 Clamp bbox values safely to [0, 1000]
            xmin = max(0, min(xmin, 1000))
            ymin = max(0, min(ymin, 1000))
            xmax = max(0, min(xmax, 1000))
            ymax = max(0, min(ymax, 1000))

            clean_text = " ".join(li.text.split())
            tokens = self.processor.tokenizer.tokenize(clean_text)
            if not tokens:
                continue

            field = mapping.get(id(li), None)
            if field in FIELD_TO_BI:
                b_label, i_label = FIELD_TO_BI[field]
            else:
                b_label, i_label = "O", "O"

            for j, tok in enumerate(tokens):
                words.append(tok)
                boxes.append([xmin, ymin, xmax, ymax])
                if field in FIELD_TO_BI:
                    labels.append(LABEL2ID[b_label if j == 0 else i_label])
                else:
                    labels.append(LABEL2ID["O"])

        # --- Sanity check for invalid labels ---
        if any(l >= len(LABEL2ID) or l < -100 for l in labels):
            print(f"⚠️ Invalid label index detected in {stem}, resetting to O.")
            labels = [LABEL2ID["O"] if (l >= len(LABEL2ID) or l < -100) else l for l in labels]

        # --- Pad or truncate safely ---
        pad_len = self.max_seq_len - len(words)
        if pad_len < 0:
            words = words[:self.max_seq_len]
            boxes = boxes[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            boxes += [[0, 0, 0, 0]] * pad_len
            words += ["[PAD]"] * pad_len
            labels += [-100] * pad_len

        # --- Encode via processor ---
        try:
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
        except Exception as e:
            print(f"❌ Encoding failed for {stem}: {e}")
            # Skip bad sample
            return self.__getitem__((idx + 1) % len(self.stems))

        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["id"] = stem

        # --- Save to cache ---
        try:
            torch.save(item, cache_path)
        except Exception:
            print(f"⚠️ Could not cache {stem}.")

        return item




