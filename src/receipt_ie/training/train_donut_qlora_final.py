# layoutlmv3_finetune_receipt.py

import os, json, torch
from datasets import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)
from PIL import Image
from tqdm import tqdm

data_root = "/kaggle/input/receipt-dataset"  # adjust path

labels = ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE", "B-ADDRESS", "I-ADDRESS", "B-TOTAL", "I-TOTAL"]
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def load_ocr_file(path):
    ocr_data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9: continue
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, parts[:8])
            text = ",".join(parts[8:]).strip()
            box = [min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)]
            ocr_data.append((text, box))
    return ocr_data

def prepare_example(img_path, entities_path, ocr_path):
    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    ocr_data = load_ocr_file(ocr_path)
    with open(entities_path, encoding="utf-8") as f:
        entities = json.load(f)

    words, boxes, labels_ = [], [], []

    for text, box in ocr_data:
        box_norm = normalize_box(box, W, H)
        label = "O"
        for key, val in entities.items():
            if val and text.lower() in val.lower():
                label = f"B-{key.upper()}" if label == "O" else f"I-{key.upper()}"
        words.append(text)
        boxes.append(box_norm)
        labels_.append(label2id.get(label, 0))

    encoding = processor(image, words, boxes=boxes, truncation=True, padding="max_length", max_length=512)
    encoding["labels"] = labels_ + [0]*(512 - len(labels_))
    return encoding

# --- Build Dataset ---
samples = []
for split in ["train", "test"]:
    img_dir = os.path.join(data_root, split, "img")
    ent_dir = os.path.join(data_root, split, "entities")
    ocr_dir = os.path.join(data_root, split, "ocr")  # ensure folder name
    for fname in tqdm(os.listdir(img_dir), desc=f"{split}"):
        stem, _ = os.path.splitext(fname)
        img_path = os.path.join(img_dir, fname)
        ent_path = os.path.join(ent_dir, stem + ".json")
        ocr_path = os.path.join(ocr_dir, stem + ".txt")
        if not os.path.exists(ent_path) or not os.path.exists(ocr_path): continue
        samples.append(prepare_example(img_path, ent_path, ocr_path))

dataset = Dataset.from_list(samples)
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, val_ds = train_test["train"], train_test["test"]

# --- Model ---
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# --- Training ---
args = TrainingArguments(
    output_dir="/kaggle/temp/outputs_layoutlmv3",
    learning_rate=3e-5,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    save_total_limit=2,
    logging_steps=25,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor,
)

trainer.train()
model.save_pretrained("/kaggle/temp/outputs_layoutlmv3/final_model")
processor.save_pretrained("/kaggle/temp/outputs_layoutlmv3/final_model")
print("✅ LayoutLMv3 fine-tuning complete.")
