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


def visualize_sample(stem):
    img_path = find_with_ext(IMG_DIR, stem)
    box_path = os.path.join(BOX_DIR, stem + ".txt")
    ent_path = os.path.join(ENT_DIR, stem + ".json")

    if not (os.path.isfile(img_path) and os.path.isfile(box_path) and os.path.isfile(ent_path)):
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
    plt.show()


def main():
    stems = [os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if len(stems) == 0:
        print("No images found.")
        return

    samples = random.sample(stems, min(N_SAMPLES, len(stems)))
    print(f"🔍 Visualizing {len(samples)} samples...")

    for s in samples:
        visualize_sample(s)


if __name__ == "__main__":
    main()
