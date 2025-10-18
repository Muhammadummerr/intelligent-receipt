#!/usr/bin/env python3
"""
build_watermark_dataset_advanced.py
-----------------------------------
Advanced synthetic watermark dataset builder (Kaggle-ready).

✅ Includes realistic watermark types:
    • text_stamp (semi-transparent overlay text)
    • blur_top (header blur)
    • black_bar (occlusion bar)
    • highlight_patch (yellow smudge)
    • ink_bleed (handwritten overlay)
    • shadow_patch (gray smudge)

✅ Auto-balances clean/watermarked classes
✅ Supports previews
✅ Works with Pillow>=10
"""

import os, random, argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------
# GLOBAL SETTINGS
# -------------------------------
DEFAULT_WM_TEXTS = [
    "CONFIDENTIAL", "SAMPLE", "DEMO", "VOID", "TEST",
    "DO NOT COPY", "COPY", "UNOFFICIAL", "TRAINING"
]

FAKE_NAMES = [
    "tan chay yee", "ahmad zainal", "lee chong", "siti nur",
    "ong kah weng", "lim boon", "mohd faiz"
]

# -------------------------------
# FONTS
# -------------------------------
def load_font(font_path: Optional[str], size: int):
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


# -------------------------------
# BASIC EFFECTS
# -------------------------------
def top_region_box(W, H, frac=0.2):
    return (int(W*0.05), int(H*0.02), int(W*0.95), int(H*frac))

def blackout_bar(img, box, color=(0,0,0)):
    out = img.copy()
    ImageDraw.Draw(out).rectangle(box, fill=color)
    return out

def local_blur(img, box, radius=10):
    region = img.crop(box).filter(ImageFilter.GaussianBlur(radius=radius))
    out = img.copy()
    out.paste(region, box)
    return out

def translucent_wash(img, color=(180,180,180), alpha=0.2):
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, color + (int(255*alpha),))
    return Image.alpha_composite(base, overlay).convert("RGB")


# -------------------------------
# ADVANCED WATERMARK EFFECTS
# -------------------------------
def overlay_text_stamp(img, text, opacity=0.15, angle=-20, font_path=None):
    W,H = img.size
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)

    font = load_font(font_path, int(min(W,H)*0.12))
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)

    txt = Image.new("RGBA", (tw+10, th+10), (255,255,255,0))
    td = ImageDraw.Draw(txt)
    td.text((5,5), text, font=font, fill=(0,0,0,255))
    txt = txt.rotate(angle, expand=1)

    step_x, step_y = tw*2, th*2
    for x in range(-step_x, W+step_x, step_x):
        for y in range(-step_y, H+step_y, step_y):
            overlay.paste(txt, (x,y), txt)

    alpha = overlay.split()[3].point(lambda p: int(p*opacity))
    overlay.putalpha(alpha)
    return Image.alpha_composite(base, overlay).convert("RGB")


def highlight_patch(img, opacity=0.25):
    """Add faint yellow highlighter stroke."""
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (255,255,0,0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(1,3)):
        x1 = random.randint(0,int(W*0.6))
        y1 = random.randint(0,int(H*0.5))
        x2 = x1 + random.randint(int(W*0.3), int(W*0.9))
        y2 = y1 + random.randint(20,60)
        draw.rectangle([x1,y1,x2,y2], fill=(255,255,0,int(255*opacity)))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def ink_bleed(img, text=None, font_path=None):
    """Handwritten overlay in semi-transparent gray."""
    if text is None:
        text = random.choice(FAKE_NAMES)
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(font_path, int(min(W,H)*0.07))
    x = random.randint(0,int(W*0.3))
    y = random.randint(0,int(H*0.2))
    draw.text((x,y), text, font=font, fill=(50,50,50,130))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def shadow_patch(img):
    """Dark smudge near top."""
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(1,3)):
        x1 = random.randint(0,int(W*0.6))
        y1 = random.randint(0,int(H*0.3))
        x2 = x1 + random.randint(int(W*0.4), int(W*0.8))
        y2 = y1 + random.randint(30,80)
        draw.rectangle([x1,y1,x2,y2], fill=(0,0,0,random.randint(30,80)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# -------------------------------
# EFFECT MIXER
# -------------------------------
ADVANCED_WM_TYPES = {
    "stamp": lambda img: overlay_text_stamp(
        img,
        text=random.choice(DEFAULT_WM_TEXTS),
        opacity=random.uniform(0.08, 0.2),
        angle=random.uniform(-25, 25),
    ),
    "blur_top": lambda img: local_blur(img, top_region_box(*img.size)),
    "black_bar": lambda img: blackout_bar(img, top_region_box(*img.size)),
    "highlight_patch": lambda img: highlight_patch(img, opacity=random.uniform(0.15, 0.3)),
    "ink_bleed": lambda img: ink_bleed(img, text=random.choice(FAKE_NAMES)),
    "shadow_patch": shadow_patch,
}


def synthesize_watermarked_image(img, rng):
    """Apply 1–3 random watermark effects."""
    effects = rng.sample(list(ADVANCED_WM_TYPES.keys()), rng.randint(1,3))
    reason_list = []
    for e in effects:
        img = ADVANCED_WM_TYPES[e](img)
        reason_list.append(e)
    return img, "+".join(effects), f"Combined effects: {', '.join(reason_list)}"


# -------------------------------
# DATASET BUILDER
# -------------------------------
def build_dataset(input_dir, output_dir, target_per_class=500,
                  val_frac=0.15, test_frac=0.15, seed=42,
                  font_path=None, preview=0):
    rng = random.Random(seed)
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in input_dir.glob("*") if p.suffix.lower() in [".jpg",".png",".jpeg"]]
    rng.shuffle(imgs)

    clean, wm = [], []
    for p in tqdm(imgs, desc="Synthesizing"):
        if len(clean)>=target_per_class and len(wm)>=target_per_class:
            break
        try:
            im = Image.open(p).convert("RGB")
        except Exception: continue
        if len(clean)<target_per_class:
            clean.append((im.copy(), p.name, "clean", "original"))
        if len(wm)<target_per_class:
            wim, typ, reason = synthesize_watermarked_image(im.copy(), rng)
            wm.append((wim, p.name, typ, reason))

    # Preview
    if preview>0:
        import matplotlib.pyplot as plt
        samples = rng.sample(wm+clean, min(preview, len(wm)+len(clean)))
        plt.figure(figsize=(12,6))
        for i,(im,name,typ,reason) in enumerate(samples):
            plt.subplot(2,(len(samples)+1)//2,i+1)
            plt.imshow(im)
            plt.axis("off")
            plt.title(typ, fontsize=10)
        plt.show()
        if input("Continue building dataset? (y/n): ").strip().lower()!="y":
            print("❌ Cancelled."); return

    # Split and write
    all_ = clean + wm
    rng.shuffle(all_)
    n=len(all_); n_val=int(n*val_frac); n_test=int(n*test_frac); n_train=n-n_val-n_test
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
    print(f"✅ Dataset complete at {output_dir}")

# -------------------------------
# CLI
# -------------------------------
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--input_dir",required=True)
    a.add_argument("--output_dir",required=True)
    a.add_argument("--target_per_class",type=int,default=400)
    a.add_argument("--val_frac",type=float,default=0.15)
    a.add_argument("--test_frac",type=float,default=0.15)
    a.add_argument("--seed",type=int,default=42)
    a.add_argument("--font_path",default=None)
    a.add_argument("--preview",type=int,default=8)
    args=a.parse_args()
    build_dataset(**vars(args))
