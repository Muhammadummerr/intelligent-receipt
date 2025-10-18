#!/usr/bin/env python3
"""
augment_watermark_dataset.py
-----------------------------
Expands your watermark detection dataset via smart augmentations.

Input directory (local or Kaggle):
    C:/Users/Umer/Pictures/watermark-dataset/
        ├── clean/
        ├── watermark/

Output:
    /kaggle/working/watermark-augmented/
        ├── train/
        │   ├── clean/
        │   ├── watermarked/
        ├── test/
        │   ├── clean/
        │   ├── watermarked/
"""

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import cv2
from sklearn.model_selection import train_test_split

# ========== CONFIG ==========
INPUT_DIR = r"C:\Users\Umer\Pictures\watermark-dataset"  # <-- local path
OUTPUT_DIR = r"C:\Users\Umer\intelligent-receipt\src\receipt_ie\data\watermark-augmented"
SPLIT_RATIO = 0.8  # 80% train, 20% test
AUG_PER_IMAGE = 3  # how many augmented copies per original
SEED = 42
random.seed(SEED)

# ========== AUGMENTATION PIPELINE ==========
augment = A.Compose([
    A.Rotate(limit=3, p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),
    A.CLAHE(p=0.3),
    A.GaussNoise(var_limit=(5, 20), p=0.3),
    A.Affine(scale=(0.98, 1.02), translate_percent=(0.01, 0.02), rotate=(-2, 2), p=0.3)
])

# ========== HELPER ==========
def ensure_dirs(base):
    for split in ["train", "test"]:
        for cls in ["clean", "watermarked"]:
            os.makedirs(os.path.join(base, split, cls), exist_ok=True)

def augment_and_save(img_path, out_dir, n_aug):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    base_name = Path(img_path).stem

    # Save original
    cv2.imwrite(os.path.join(out_dir, f"{base_name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Augmentations
    for i in range(n_aug):
        aug_img = augment(image=img)["image"]
        out_path = os.path.join(out_dir, f"{base_name}_aug{i+1}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

# ========== MAIN ==========
def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    ensure_dirs(output_dir)

    all_classes = [("clean", "clean"), ("watermark", "watermarked")]

    for src_cls, dst_cls in all_classes:
        src_dir = input_dir / src_cls
        if not src_dir.exists():
            print(f"⚠️ Missing {src_dir}")
            continue

        all_imgs = [p for p in src_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        train_imgs, test_imgs = train_test_split(all_imgs, train_size=SPLIT_RATIO, random_state=SEED)

        for split_name, subset in [("train", train_imgs), ("test", test_imgs)]:
            split_dir = output_dir / split_name / dst_cls
            print(f"Processing {src_cls} -> {split_name} ({len(subset)} images)...")
            for img_path in tqdm(subset):
                try:
                    augment_and_save(img_path, split_dir, AUG_PER_IMAGE)
                except Exception as e:
                    print(f"❌ Error on {img_path}: {e}")

    print(f"\n✅ Augmented dataset created at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
