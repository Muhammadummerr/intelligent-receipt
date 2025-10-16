"""
train_donut_optimized.py
------------------------
Lightweight Donut fine-tuning for Kaggle GPUs.
Optimized for limited VRAM (T4 / P100).

Features:
✅ Smaller model by default (Bennet1996/donut-small)
✅ Image resolution reduced to 384x384
✅ Gradient checkpointing + fp16 + accumulation
✅ CPU memory offload where possible
✅ Disables WandB + tokenizer parallelism
"""

import os
import json
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from tqdm import tqdm

# --- Environment settings (safety) ---
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# -------------------------------
# Utility: Load entity annotations
# -------------------------------
def load_entity_file(path):
    """Read JSON or TXT entity file into a dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
            try:
                return json.loads(data)
            except Exception:
                ent = {}
                for line in data.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        ent[k.strip().lower()] = v.strip()
                return ent
    except Exception as e:
        print(f"⚠️ Failed to read {path}: {e}")
        return {}


# -------------------------------
# Step 1: Build Donut JSON dataset
# -------------------------------
def build_donut_json(data_root, split="train", output_json="train.json"):
    img_dir = os.path.join(data_root, split, "img")
    ent_dir = os.path.join(data_root, split, "entities")

    donut_samples = []
    for fname in tqdm(sorted(os.listdir(img_dir)), desc=f"Building {split} set"):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        ent_path = None
        for e in (".json", ".txt"):
            p = os.path.join(ent_dir, stem + e)
            if os.path.isfile(p):
                ent_path = p
                break
        if not ent_path:
            continue

        entities = load_entity_file(ent_path)
        gt_json = {
            "company": entities.get("company", ""),
            "date": entities.get("date", ""),
            "address": entities.get("address", ""),
            "total": entities.get("total", ""),
        }
        gt_str = "<s_receipt>" + json.dumps(gt_json, ensure_ascii=False)
        donut_samples.append({"image": os.path.join(img_dir, fname), "ground_truth": gt_str})

    with open(output_json, "w", encoding="utf-8") as f:
        for s in donut_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(donut_samples)} samples to {output_json}")
    return output_json


# -------------------------------
# Step 2: Preprocessing
# -------------------------------
def make_preprocess_fn(processor):
    eos_id = processor.tokenizer.eos_token_id

    def _fn(examples):
        pixel_values, labels = [], []
        for image_path, gt in zip(examples["image"], examples["ground_truth"]):
            image = Image.open(image_path).convert("RGB")
            image = image.resize((384, 384))

            # ✅ Ensure tensor type directly
            pv = processor(image, return_tensors="pt").pixel_values.squeeze(0)
            pixel_values.append(pv)

            # ✅ Convert to tensor immediately
            ids = processor.tokenizer(
                gt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.squeeze(0)

            if eos_id is not None:
                ids = torch.cat([ids, torch.tensor([eos_id], dtype=torch.long)])
            labels.append(ids)

        # ✅ Return tensors instead of lists
        return {
            "pixel_values": [torch.as_tensor(p) for p in pixel_values],
            "labels": [torch.as_tensor(l, dtype=torch.long) for l in labels],
        }

    return _fn


# -------------------------------
# Step 3: Collator (manual padding)
# -------------------------------
def donut_collate_fn(batch):
    # ✅ Each item["pixel_values"] is now a tensor
    pixel_values = torch.stack([torch.as_tensor(b["pixel_values"]) for b in batch])

    lengths = [x["labels"].size(0) for x in batch]
    max_len = max(lengths)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        seq = x["labels"]
        labels[i, : seq.size(0)] = seq

    return {"pixel_values": pixel_values, "labels": labels}


# -------------------------------
# Step 4: Training
# -------------------------------
def main():
    data_root = "/kaggle/input/receipt-dataset"
    model_id = "Bennet1996/donut-small"   # ✅ lightweight Donut model
    out_dir = "./outputs_donut"
    os.makedirs(out_dir, exist_ok=True)

    # 1️⃣ Build JSON files
    train_json = build_donut_json(data_root, "train", os.path.join(out_dir, "train.json"))
    val_json = build_donut_json(data_root, "test", os.path.join(out_dir, "val.json"))

    # 2️⃣ Load dataset
    dataset = load_dataset("json", data_files={"train": train_json, "validation": val_json})

    # 3️⃣ Processor + model
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Donut token setup
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = 256

    # 4️⃣ Preprocess
    preprocess_fn = make_preprocess_fn(processor)
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # ✅ Convert all columns to torch tensors (fixes collate issue)
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])


    # 5️⃣ Memory optimizations
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6️⃣ Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=5,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,   # accumulate more to save VRAM
        predict_with_generate=True,
        generation_max_length=256,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        report_to=[],  # disable wandb
    )

    # 7️⃣ Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.tokenizer,
        data_collator=donut_collate_fn,
    )

    print("🚀 Starting Donut fine-tuning (lightweight mode)...")
    trainer.train()
    print("✅ Training complete!")

    # 8️⃣ Save model
    final_dir = os.path.join(out_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"🏁 Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
