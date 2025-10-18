#!/usr/bin/env python3
"""
build_watermark_dataset_occlusion.py
------------------------------------
Advanced watermark + occlusion dataset builder for receipt classification.

✅ Generates realistic faint marks, occlusion bars, and smudges.
✅ Simulates real-world censorship / watermarks over text.
✅ Auto-balanced clean vs watermarked dataset for ViT classifier training.
"""

import os, random, argparse
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
DEFAULT_WM_TEXTS = [
    "CONFIDENTIAL", "SAMPLE", "VOID", "TRAINING", "DO NOT COPY", "UNOFFICIAL"
]
FAKE_NAMES = [
    "tan chay yee", "ahmad zainal", "lim boon", "ong kah weng", "siti nur", "mohd faiz"
]

# ---------------------------
# FONTS
# ---------------------------
def load_font(font_path: Optional[str], size: int):
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

# ---------------------------
# EFFECTS
# ---------------------------
def overlay_text_stamp(img, text, opacity=0.1, angle=-20, font_path=None):
    """Semi-transparent repeating text watermark."""
    W,H = img.size
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(font_path, int(min(W,H)*0.12))

    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    txt = Image.new("RGBA", (tw, th), (255,255,255,0))
    td = ImageDraw.Draw(txt)
    td.text((0,0), text, font=font, fill=(0,0,0,255))
    txt = txt.rotate(angle, expand=1)
    for x in range(-tw, W+tw, int(tw*2)):
        for y in range(-th, H+th, int(th*2)):
            overlay.paste(txt, (x,y), txt)
    alpha = overlay.split()[3].point(lambda p: int(p*opacity*255))
    overlay.putalpha(alpha)
    return Image.alpha_composite(base, overlay).convert("RGB")


def small_occlusion_patch(img, opacity=0.7):
    """Small black or white patches hiding parts of text."""
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(1, 4)):
        w = random.randint(int(W*0.05), int(W*0.20))
        h = random.randint(int(H*0.02), int(H*0.06))
        x = random.randint(0, W-w)
        y = random.randint(int(H*0.05), int(H*0.8))
        color = random.choice([(0,0,0,int(255*opacity)), (255,255,255,int(255*opacity))])
        draw.rectangle([x,y,x+w,y+h], fill=color)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def black_line_occlusion(img, thickness=4):
    """Draw 1–3 thin horizontal black lines across text area."""
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(1,3)):
        y = random.randint(int(H*0.15), int(H*0.7))
        draw.rectangle([0, y, W, y+thickness], fill=(0,0,0,random.randint(150,220)))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def faint_shadow(img):
    """Grayish smudge over partial area."""
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(1,2)):
        x1 = random.randint(0,int(W*0.6))
        y1 = random.randint(0,int(H*0.4))
        x2 = x1 + random.randint(int(W*0.3), int(W*0.9))
        y2 = y1 + random.randint(30,60)
        draw.rectangle([x1,y1,x2,y2], fill=(0,0,0,random.randint(30,80)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=8))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def ink_name_overlay(img, font_path=None):
    """Fake handwritten name overlay (like signature or faint watermark)."""
    text = random.choice(FAKE_NAMES)
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(font_path, int(min(W,H)*0.07))
    x = random.randint(int(W*0.05), int(W*0.4))
    y = random.randint(int(H*0.1), int(H*0.3))
    draw.text((x,y), text, font=font, fill=(0,0,0,130))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

# ---------------------------
# MIXER
# ---------------------------
WM_EFFECTS = {
    "stamp": lambda img: overlay_text_stamp(img, random.choice(DEFAULT_WM_TEXTS)),
    "occlusion_patch": small_occlusion_patch,
    "black_line": black_line_occlusion,
    "shadow": faint_shadow,
    "ink_name": ink_name_overlay,
}

def synthesize_watermarked_image(img, rng):
    """Apply 1–2 random watermark/occlusion effects."""
    effects = rng.sample(list(WM_EFFECTS.keys()), rng.randint(1,2))
    reason_list = []
    for e in effects:
        img = WM_EFFECTS[e](img)
        reason_list.append(e)
    return img, "+".join(effects), f"Applied: {', '.join(reason_list)}"

# ---------------------------
# DATASET BUILDER
# ---------------------------
def build_dataset(input_dir, output_dir, target_per_class=500,
                  val_frac=0.15, test_frac=0.15, seed=42, font_path=None):
    rng = random.Random(seed)
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in input_dir.glob("*") if p.suffix.lower() in [".jpg",".png",".jpeg"]]
    rng.shuffle(imgs)

    clean, wm = [], []
    for p in tqdm(imgs, desc="Synthesizing"):
        if len(clean) >= target_per_class and len(wm) >= target_per_class:
            break
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        if len(clean) < target_per_class:
            clean.append((im.copy(), p.name, "clean", "original"))
        if len(wm) < target_per_class:
            wim, typ, reason = synthesize_watermarked_image(im.copy(), rng)
            wm.append((wim, p.name, typ, reason))

    all_ = clean + wm
    rng.shuffle(all_)
    n=len(all_)
    n_val=int(n*val_frac); n_test=int(n*test_frac); n_train=n-n_val-n_test
    splits={"train":all_[:n_train],"val":all_[n_train:n_train+n_val],"test":all_[n_train+n_val:]}

    for split in splits:
        for cls in ["clean","watermarked"]:
            (output_dir/split/cls).mkdir(parents=True, exist_ok=True)

    rows=[]
    for split,items in splits.items():
        for im,name,typ,reason in items:
            label="clean" if typ=="clean" else "watermarked"
            out=output_dir/split/label/f"{split}_{name}"
            out=out.with_suffix(".jpg")
            im.save(out,quality=90)
            rows.append({"image":out.name,"split":split,"label":label,"type":typ,"reason":reason})
    pd.DataFrame(rows).to_csv(output_dir/"metadata.csv",index=False)
    print(f"✅ Dataset created at {output_dir} with {len(rows)} images")

# ---------------------------
# CLI
# ---------------------------
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--input_dir",required=True)
    a.add_argument("--output_dir",required=True)
    a.add_argument("--target_per_class",type=int,default=500)
    a.add_argument("--val_frac",type=float,default=0.15)
    a.add_argument("--test_frac",type=float,default=0.15)
    a.add_argument("--seed",type=int,default=42)
    a.add_argument("--font_path",default=None)
    args=a.parse_args()
    build_dataset(**vars(args))
