import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from receipt_ie.data.boxes import parse_box_file, sort_reading_order
from receipt_ie.data.align import load_entities, assign_lines_to_fields
from receipt_ie.utils.text import normalize_spaces


# === Configuration ===
IMG_DIR = "/kaggle/input/receipt-dataset/train/img"
BOX_DIR = "/kaggle/input/receipt-dataset/train/box"
ENT_DIR = "/kaggle/input/receipt-dataset/train/entities"

# how many receipts to visualize
N_SAMPLES = 10

# color per field
FIELD_COLORS = {
    "COMPANY": "royalblue",
    "ADDRESS": "limegreen",
    "DATE": "orange",
    "TOTAL": "crimson",
}


def find_with_ext(dirpath, stem, exts=(".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
    for e in exts:
        p = os.path.join(dirpath, stem + e)
        if os.path.isfile(p):
            return p
    return None


def find_with_ext(dirpath, stem, exts):
    for e in exts:
        p = os.path.join(dirpath, stem + e)
        if os.path.isfile(p):
            return p
    return None

def visualize_sample(stem):
    img_path = find_with_ext(IMG_DIR, stem, [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"])
    box_path = find_with_ext(BOX_DIR, stem, [".txt", ".TXT"])
    ent_path = find_with_ext(ENT_DIR, stem, [".json", ".JSON", ".txt", ".TXT"])

    if not (img_path and box_path and ent_path):
        print(f"⚠️ Missing file(s) for {stem}")
        return


    # load image + OCR boxes + GT entities
    image = Image.open(img_path).convert("RGB")
    lines = sort_reading_order(parse_box_file(box_path))
    gt = load_entities(ent_path)

    # get predicted mapping (our alignment)
    mapping = assign_lines_to_fields(lines, gt)

    # visualize
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(image)
    ax.set_title(f"Receipt: {stem}", fontsize=14)

    for li in lines:
        xmin, ymin, xmax, ymax = li.aabb
        field = mapping.get(id(li))
        color = FIELD_COLORS.get(field, "gray")
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        label = f"{field or 'UNK'}: {normalize_spaces(li.text)[:40]}"
        ax.text(xmin, ymin - 3, label, color=color, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    plt.axis("off")
    plt.tight_layout()

    out_dir = "/kaggle/working/audit_vis"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}.jpg")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved visualization to {out_path}")


def main():
    # collect only stems that have all 3 files present
    img_stems = {os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    box_stems = {os.path.splitext(f)[0] for f in os.listdir(BOX_DIR) if f.lower().endswith(".txt")}
    ent_stems = {os.path.splitext(f)[0] for f in os.listdir(ENT_DIR) if f.lower().endswith((".json", ".txt"))}

    valid_stems = sorted(list(img_stems & box_stems & ent_stems))
    print(f"✅ Found {len(valid_stems)} valid receipts with image+box+entity")

    if not valid_stems:
        print("No valid triplets found.")
        return

    samples = random.sample(valid_stems, min(N_SAMPLES, len(valid_stems)))
    print(f"🔍 Visualizing {len(samples)} samples...")

    for s in samples:
        visualize_sample(s)

if __name__ == "__main__":
    main()
