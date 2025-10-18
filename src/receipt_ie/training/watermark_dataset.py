#!/usr/bin/env python3
"""
build_watermark_dataset.py

Usage (example):
python build_watermark_dataset.py \
  --input_dir /kaggle/input/clean-receipts \
  --output_dir /kaggle/working/receipt-watermark-dataset \
  --target_per_class 800 \
  --val_frac 0.15 \
  --test_frac 0.15 \
  --seed 42 \
  --use_box_dir /kaggle/input/receipt-boxes  # optional
"""

import os
import random
import argparse
import math
import csv
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import albumentations as A
import pandas as pd

# -----------------------
# Default watermark texts (generic) — do NOT include real company names here
# -----------------------
DEFAULT_WM_TEXTS = [
    "CONFIDENTIAL", "SAMPLE", "DEMO", "VOID", "TEST", "DO NOT COPY",
    "COPY", "UNOFFICIAL", "FOR TRAINING ONLY", "NOT FOR RE-SALE"
]

# -----------------------
# Helpers: fonts
# -----------------------
def load_font(font_path: Optional[str], size: int):
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
        # fallback to default PIL font (monospace)
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

# -----------------------
# Watermark generators
# -----------------------
def overlay_text_stamp(
    img: Image.Image,
    text: str,
    opacity: float = 0.18,
    angle: float = -20,
    scale: float = 1.0,
    font_path: Optional[str] = None,
    fontsize: Optional[int] = None,
    fill=(0, 0, 0),
) -> Image.Image:
    """Large rotated translucent text across the image (stamp)."""
    W, H = img.size
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # estimate font size relative to image
    if fontsize is None:
        fontsize = max(20, int(min(W, H) * 0.12 * scale))
    font = load_font(font_path, fontsize)

    # create repeated tiled text so it crosses image
    text_w, text_h = draw.textsize(text, font=font)
    # create a text image
    txt_img = Image.new("RGBA", (text_w + 10, text_h + 10), (255,255,255,0))
    td = ImageDraw.Draw(txt_img)
    td.text((5, 5), text, font=font, fill=fill + (255,))

    # rotate and tile
    txt_img = txt_img.rotate(angle, expand=1)
    tx_w, tx_h = txt_img.size

    # tile across center region
    step_x = int(tx_w * 1.2)
    step_y = int(tx_h * 1.8)
    start_x = -step_x
    start_y = H // 4 - tx_h // 2
    for x in range(start_x, W + step_x, step_x):
        for y in range(start_y, H + step_y, step_y):
            overlay.paste(txt_img, (x, y), txt_img)

    # apply opacity
    alpha = overlay.split()[3].point(lambda p: int(p * opacity))
    overlay.putalpha(alpha)
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

def translucent_box_erase(
    img: Image.Image,
    box: Tuple[int,int,int,int],
    blur_radius: int = 7,
    fill_color=(255,255,255),
    alpha: float = 0.8,
) -> Image.Image:
    """Draw a translucent filled box and optionally blur inside to hide text."""
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    xmin, ymin, xmax, ymax = box
    draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color + (int(255*alpha),))
    merged = Image.alpha_composite(base, overlay).convert("RGB")
    # blur region
    region = merged.crop(box).filter(ImageFilter.GaussianBlur(blur_radius))
    merged.paste(region, box)
    return merged

def local_gaussian_blur(
    img: Image.Image,
    box: Tuple[int,int,int,int],
    radius: int = 10
) -> Image.Image:
    """Apply Gaussian blur to a local bounding box region."""
    out = img.copy()
    region = out.crop(box).filter(ImageFilter.GaussianBlur(radius=radius))
    out.paste(region, box)
    return out

def blackout_bar(
    img: Image.Image,
    box: Tuple[int,int,int,int],
    color=(0,0,0),
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(box, fill=color)
    return out

def scribble_line(
    img: Image.Image,
    box: Tuple[int,int,int,int],
    strokes: int = 3,
    width_range=(3, 10),
    color=(0,0,0)
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    xmin, ymin, xmax, ymax = box
    for s in range(strokes):
        y = random.randint(ymin, ymax)
        x0 = random.randint(xmin, max(xmin+1, xmin + (xmax-xmin)//6))
        x1 = random.randint(max(x0+1, xmax - (xmax-xmin)//6), xmax)
        width = random.randint(*width_range)
        draw.line([(x0,y), (x1, y + random.randint(-5,5))], fill=color, width=width)
    return out

def translucent_wash(img: Image.Image, color=(200,200,200), alpha=0.18) -> Image.Image:
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, color + (int(255*alpha),))
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

# -----------------------
# Utilities: target boxes heuristics
# -----------------------
def top_region_box(W: int, H: int, frac_height: float = 0.18) -> Tuple[int,int,int,int]:
    """Return a box roughly covering the top of receipt (company area)."""
    xmin = int(W * 0.05)
    xmax = int(W * 0.95)
    ymin = int(H * 0.02)
    ymax = int(H * frac_height)  # top ~18%
    return (xmin, ymin, xmax, ymax)

def center_region_box(W: int, H: int, wfrac=0.6, hfrac=0.08) -> Tuple[int,int,int,int]:
    xmin = int(W*(1-wfrac)/2)
    xmax = int(W*(1+(wfrac))/2)
    ymin = int(H*0.45 - H*hfrac/2)
    ymax = int(H*0.45 + H*hfrac/2)
    return (xmin, ymin, xmax, ymax)

# -----------------------
# Main generation logic
# -----------------------
WATERMARK_TYPES = [
    "stamp_text",
    "top_blur",
    "erase_box",
    "black_bar_top",
    "scribble_top",
    "translucent_wash",
]

def synthesize_watermarked_image(
    img: Image.Image,
    wm_texts: List[str],
    font_path: Optional[str],
    use_boxes: Optional[Dict[str, Tuple[int,int,int,int]]],
    rng: random.Random,
) -> Tuple[Image.Image, str, str]:
    """Return (image, watermark_type, reason)."""
    W, H = img.size
    wm_type = rng.choice(WATERMARK_TYPES)
    reason = ""
    out = img.copy()

    # pick a random watermark text
    wm_text = rng.choice(wm_texts)

    if wm_type == "stamp_text":
        opacity = rng.uniform(0.10, 0.28)
        angle = rng.uniform(-30, 30)
        scale = rng.uniform(0.8, 1.2)
        fontsize = max(20, int(min(W,H)*0.12*scale))
        out = overlay_text_stamp(out, wm_text, opacity=opacity, angle=angle, scale=scale, font_path=font_path, fontsize=fontsize)
        reason = f"Large translucent stamp '{wm_text}'"
    elif wm_type == "top_blur":
        # blur top region or OCR-provided company box
        box = use_boxes.get("company") if use_boxes and "company" in use_boxes else top_region_box(W,H)
        radius = rng.randint(8, 25)
        out = local_gaussian_blur(out, box, radius=radius)
        reason = f"Local blur radius {radius}"
    elif wm_type == "erase_box":
        box = use_boxes.get("company") if use_boxes and "company" in use_boxes else top_region_box(W,H)
        out = translucent_box_erase(out, box, blur_radius=rng.randint(5,15), alpha=rng.uniform(0.6,0.95))
        reason = "Translucent white erase over region"
    elif wm_type == "black_bar_top":
        # black bar across top
        box = use_boxes.get("company") if use_boxes and "company" in use_boxes else top_region_box(W,H)
        # extend slightly horizontally
        xmin,ymin,xmax,ymax = box
        pad = int((xmax-xmin)*rng.uniform(0.02,0.2))
        box2 = (max(0,xmin-pad), ymin, min(W,xmax+pad), min(H,ymax+(ymax-ymin)//2))
        out = blackout_bar(out, box2, color=(0,0,0))
        reason = "Black bar occlusion"
    elif wm_type == "scribble_top":
        box = use_boxes.get("company") if use_boxes and "company" in use_boxes else top_region_box(W,H)
        out = scribble_line(out, box, strokes=rng.randint(2,6), width_range=(2,12))
        reason = "Scribbled lines over top region"
    elif wm_type == "translucent_wash":
        color = (rng.randint(100,230), rng.randint(100,230), rng.randint(100,230))
        alpha = rng.uniform(0.08, 0.32)
        out = translucent_wash(out, color=color, alpha=alpha)
        reason = "Semi-transparent color wash"
    else:
        # Fallback to stamp
        out = overlay_text_stamp(out, wm_text, opacity=0.18, angle=-15, scale=1.0, font_path=font_path)
        reason = f"Fallback stamp '{wm_text}'"

    return out, wm_type, reason

# -----------------------
# Dataset builder
# -----------------------
def build_dataset(
    input_dir: str,
    output_dir: str,
    target_per_class: int = 800,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    wm_texts: Optional[List[str]] = None,
    font_path: Optional[str] = None,
    use_box_dir: Optional[str] = None,
):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    wm_texts = wm_texts or DEFAULT_WM_TEXTS

    input_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # scan images
    exts = {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}
    all_images = [p for p in input_dir.glob("*") if p.suffix.lower() in exts]
    if not all_images:
        raise RuntimeError(f"No images found in {input_dir}")

    # optional boxes: expects one .txt per image with format same as earlier (x1,y1,...,text)
    box_map = {}
    if use_box_dir:
        box_dir = Path(use_box_dir)
        for imgp in all_images:
            stem = imgp.stem
            candidate = box_dir / f"{stem}.txt"
            if candidate.exists():
                # parse simplistic: take top-most lines to estimate company/address boxes
                try:
                    lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
                    # use first non-empty line bounding box
                    for ln in lines[:10]:
                        parts = ln.split(",")
                        if len(parts) >= 9:
                            coords = list(map(int, parts[:8]))
                            xmin = min(coords[0],coords[2],coords[4],coords[6])
                            ymin = min(coords[1],coords[3],coords[5],coords[7])
                            xmax = max(coords[0],coords[2],coords[4],coords[6])
                            ymax = max(coords[1],coords[3],coords[5],coords[7])
                            # heuristically treat first line as company
                            box_map[imgp.name] = {"company": (xmin,ymin,xmax,ymax)}
                            break
                except Exception:
                    pass

    # prepare directories
    for split in ["train","val","test"]:
        for cls in ["clean","watermarked"]:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    metadata = []
    # target counts
    target_clean = target_per_class
    target_wm = target_per_class

    # shuffle input images
    imgs = list(all_images)
    rng.shuffle(imgs)

    # Strategy: for each input image, produce one clean copy and one (or more) watermarked copies until targets met
    clean_count = 0
    wm_count = 0
    idx = 0

    # simple splitting strategy: we will produce total_count per class then split into train/val/test by fractions
    produced = []

    while (clean_count < target_clean) or (wm_count < target_wm):
        img_path = imgs[idx % len(imgs)]
        idx += 1
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Clean copy (only if we still need)
        if clean_count < target_clean:
            produced.append({
                "image": img.copy(),
                "label": "clean",
                "orig_name": img_path.name,
                "watermark_type": "",
                "reason": "original_clean",
            })
            clean_count += 1

        # Watermarked copy
        if wm_count < target_wm:
            # use box_map for this image if present
            boxes = box_map.get(img_path.name, None)
            wm_img, wm_type, reason = synthesize_watermarked_image(img.copy(), wm_texts, font_path, boxes, rng)
            produced.append({
                "image": wm_img,
                "label": "watermarked",
                "orig_name": img_path.name,
                "watermark_type": wm_type,
                "reason": reason,
            })
            wm_count += 1

    # Now split produced list into train/val/test
    rng.shuffle(produced)
    n = len(produced)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    splits = (
        ("train", produced[:n_train]),
        ("val", produced[n_train:n_train+n_val]),
        ("test", produced[n_train+n_val:]),
    )

    print(f"Produced total images: {n}  -> train {n_train}, val {n_val}, test {n_test}")
    # write images and metadata
    meta_rows = []
    for split_name, items in splits:
        for i, item in enumerate(tqdm(items, desc=f"Writing {split_name}")):
            base_name = f"{split_name}_{i:05d}_{item['label']}_{item['orig_name']}"
            out_path = out_dir / split_name / item['label'] / base_name
            # ensure extension .jpg
            out_path = out_path.with_suffix(".jpg")
            try:
                item["image"].save(str(out_path), quality=90)
            except Exception as e:
                # fallback with OpenCV write
                arr = cv2.cvtColor(np.array(item["image"]), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), arr)

            meta_rows.append({
                "image": str(out_path.name),
                "split": split_name,
                "label": item["label"],
                "orig_image": item["orig_name"],
                "watermark_type": item.get("watermark_type", ""),
                "reason": item.get("reason", ""),
            })

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_dir / "metadata.csv", index=False)
    print("Dataset generation complete. Metadata saved to", out_dir / "metadata.csv")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Folder with clean receipt images")
    p.add_argument("--output_dir", required=True, help="Where to write dataset")
    p.add_argument("--target_per_class", type=int, default=800, help="Images per class (clean/watermarked)")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--font_path", default=None, help="Optional TTF font for watermark text")
    p.add_argument("--use_box_dir", default=None, help="Optional folder containing OCR box txt files to guide watermark placement")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_per_class=args.target_per_class,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        wm_texts=DEFAULT_WM_TEXTS,
        font_path=args.font_path,
        use_box_dir=args.use_box_dir,
    )
