"""
train_donut.py
---------------
Fine-tunes Donut (OCR-free document understanding transformer)
for key information extraction from receipts.

Dataset structure:
  root/train/img/*.jpg
  root/train/entities/*.json or .txt
  root/test/img/*.jpg
  root/test/entities/*.json or .txt
"""

import os
import json
import torch
from PIL import Image
from datasets import load_dataset, DatasetDict
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from tqdm import tqdm


# ---------------------------------------
# Utility: Load entity JSONs or TXT
# ---------------------------------------
def load_entity_file(path):
    """Read JSON or TXT entity file into a dict"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
            try:
                return json.loads(data)
            except Exception:
                # fallback if it's line-based "key:value"
                ent = {}
                for line in data.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        ent[k.strip().lower()] = v.strip()
                return ent
    except Exception as e:
        print(f"⚠️ Failed to read {path}: {e}")
        return {}

from transformers import DonutProcessor

# Custom data collator for Donut
def donut_collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.nn.utils.rnn.pad_sequence(
        [x["labels"] for x in batch],
        batch_first=True,
        padding_value=-100,
    )
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


# ---------------------------------------
# Step 1: Build Dataset JSONs for Donut
# ---------------------------------------
def build_donut_json(data_root, split="train", output_json="train.json"):
    img_dir = os.path.join(data_root, split, "img")
    ent_dir = os.path.join(data_root, split, "entities")

    donut_samples = []
    for fname in tqdm(os.listdir(img_dir), desc=f"Building {split} set"):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        ent_path = None
        for e in [".json", ".txt"]:
            path = os.path.join(ent_dir, stem + e)
            if os.path.isfile(path):
                ent_path = path
                break
        if not ent_path:
            continue

        entities = load_entity_file(ent_path)
        company = entities.get("company", "")
        date = entities.get("date", "")
        address = entities.get("address", "")
        total = entities.get("total", "")

        gt_json = {
            "company": company,
            "date": date,
            "address": address,
            "total": total,
        }
        gt_str = "<s_receipt>" + json.dumps(gt_json, ensure_ascii=False)

        donut_samples.append({
            "image": os.path.join(img_dir, fname),
            "ground_truth": gt_str
        })

    with open(output_json, "w", encoding="utf-8") as f:
        for s in donut_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(donut_samples)} samples to {output_json}")
    return output_json


# ---------------------------------------
# Step 2: Preprocessing
# ---------------------------------------
def preprocess_function(examples, processor):
    pixel_values = []
    labels = []
    for image_path, gt in zip(examples["image"], examples["ground_truth"]):
        image = Image.open(image_path).convert("RGB")
        pixel_values.append(processor(image, return_tensors="pt").pixel_values.squeeze())

        text = processor.tokenizer(
            gt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze()
        labels.append(text)

    return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------
# Step 3: Main Training Loop
# ---------------------------------------
def main():
    data_root = "/kaggle/input/receipt-dataset"
    model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
    out_dir = "./outputs_donut"

    os.makedirs(out_dir, exist_ok=True)

    # Step 1: build JSON files
    train_json = build_donut_json(data_root, "train", os.path.join(out_dir, "train.json"))
    val_json = build_donut_json(data_root, "test", os.path.join(out_dir, "val.json"))

    # Step 2: load dataset
    dataset = load_dataset("json", data_files={"train": train_json, "validation": val_json})
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Step 3: preprocess
    dataset = dataset.map(
        lambda ex: preprocess_function(ex, processor),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    torch.cuda.empty_cache()
    model.gradient_checkpointing_enable()
    # Step 4: training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=5,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        predict_with_generate=True,
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        data_collator=donut_collate_fn,
    )

    # Step 5: trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.tokenizer,
        data_collator=donut_collate_fn,
    )

    print("🚀 Starting Donut fine-tuning...")
    trainer.train()
    print("✅ Training complete!")

    model.save_pretrained(os.path.join(out_dir, "final_model"))
    processor.save_pretrained(os.path.join(out_dir, "final_model"))
    print(f"🏁 Saved fine-tuned model to {out_dir}/final_model")


if __name__ == "__main__":
    main()
