#!/usr/bin/env python3
"""
build_watermark_dataset_advanced_realistic.py
---------------------------------------------
Advanced synthetic watermark dataset builder (Kaggle-ready, realism enhanced).

✅ Includes realistic watermark types:
    • text_stamp (semi-transparent overlay text)
    • blur_top (header blur)
    • black_bar (occlusion bar)
    • highlight_patch (yellow smudge)
    • ink_bleed (handwritten overlay)
    • shadow_patch (gray smudge)
✅ Adds post-effects for realism:
    • JPEG compression
    • scan noise + contrast jitter
    • faint opacity randomization
    • slight perspective warp
✅ Auto-balances clean/watermarked classes
✅ Optionally shows preview grid
"""

import os, random, argparse, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------
# SETTINGS
# -------------------------------
DEFAULT_WM_TEXTS = [
    "CONFIDENTIAL", "SAMPLE", "DEMO", "VOID", "TEST",
    "DO NOT COPY", "COPY", "UNOFFICIAL", "TRAINING"
]

FAKE_NAMES = [
    "mohd faiz", "lee chong", "ong kah weng", "siti nur", "ahmad zainal"
]

# -------------------------------
# UTILS
# -------------------------------
def load_font(font_path, size):
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    except Exception:
        pass
    return ImageFont.load_default()

def top_region_box(W, H, frac=0.2):
    return (int(W*0.05), int(H*0.02), int(W*0.95), int(H*frac))

def blackout_bar(img, box):
    out = img.copy()
    ImageDraw.Draw(out).rectangle(box, fill=(0,0,0))
    return out

def local_blur(img, box, radius=8):
    region = img.crop(box).filter(ImageFilter.GaussianBlur(radius))
    out = img.copy(); out.paste(region, box); return out

def add_scan_noise(img, intensity=0.03):
    arr = np.array(img).astype(np.float32)/255.0
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    arr = (arr*255).astype(np.uint8)
    return Image.fromarray(arr)

def add_jpeg_artifacts(img, quality=40):
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# -------------------------------
# ADVANCED WATERMARKS
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
    if text is None:
        text = random.choice(FAKE_NAMES)
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(font_path, int(min(W,H)*0.07))
    x = random.randint(0,int(W*0.3))
    y = random.randint(0,int(H*0.2))
    draw.text((x,y), text, font=font, fill=(50,50,50,120))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

def shadow_patch(img):
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(1,2)):
        x1 = random.randint(0,int(W*0.6))
        y1 = random.randint(0,int(H*0.3))
        x2 = x1 + random.randint(int(W*0.4), int(W*0.8))
        y2 = y1 + random.randint(30,80)
        draw.rectangle([x1,y1,x2,y2], fill=(0,0,0,random.randint(20,70)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=8))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

ADVANCED_WM_TYPES = {
    "stamp": lambda img: overlay_text_stamp(
        img,
        text=random.choice(DEFAULT_WM_TEXTS),
        opacity=random.uniform(0.05, 0.15),
        angle=random.uniform(-25, 25),
    ),
    "blur_top": lambda img: local_blur(img, top_region_box(*img.size)),
    "black_bar": lambda img: blackout_bar(img, top_region_box(*img.size)),
    "highlight_patch": lambda img: highlight_patch(img, opacity=random.uniform(0.12, 0.25)),
    "ink_bleed": lambda img: ink_bleed(img, text=random.choice(FAKE_NAMES)),
    "shadow_patch": shadow_patch,
}

# -------------------------------
# REALISM POSTPROCESSING
# -------------------------------
def realism_postprocess(img):
    """Apply faint global perturbations after watermarking."""
    if random.random()<0.3:
        img = add_scan_noise(img, intensity=random.uniform(0.01, 0.04))
    if random.random()<0.3:
        img = add_jpeg_artifacts(img, quality=random.randint(40,70))
    if random.random()<0.3:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))
    if random.random()<0.2:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))
    return img

# -------------------------------
# SYNTHESIS
# -------------------------------
def synthesize_watermarked_image(img, rng):
    """Apply 1–3 watermark effects with realism jitter."""
    effects = rng.sample(list(ADVANCED_WM_TYPES.keys()), rng.randint(1,3))
    reasons=[]
    for e in effects:
        img = ADVANCED_WM_TYPES[e](img)
        reasons.append(e)
    img = realism_postprocess(img)
    return img, "+".join(effects), f"Combined effects: {', '.join(reasons)}"

# -------------------------------
# DATASET BUILDER
# -------------------------------
def build_dataset(input_dir, output_dir, target_per_class=500,
                  val_frac=0.15, test_frac=0.15, seed=42,
                  font_path=None, preview=4):
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
        except Exception:
            continue
        if len(clean)<target_per_class:
            clean.append((im.copy(), p.name, "clean", "original"))
        if len(wm)<target_per_class:
            wim, typ, reason = synthesize_watermarked_image(im.copy(), rng)
            wm.append((wim, p.name, typ, reason))

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
    print(f"✅ Dataset built at {output_dir}")

    # optional preview
    if preview>0:
        fig, axes = plt.subplots(2, preview//2, figsize=(12,5))
        for ax, (_,name,typ,_) in zip(axes.flatten(), all_[:preview]):
            img = Image.open(output_dir/"train"/("watermarked" if typ!="clean" else "clean")/f"train_{name}")
            ax.imshow(img); ax.axis("off"); ax.set_title(typ)
        plt.tight_layout(); plt.show()

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
    a.add_argument("--preview",type=int,default=6)
    args=a.parse_args()
    build_dataset(**vars(args))
