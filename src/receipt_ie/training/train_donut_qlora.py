"""
train_donut_qlora.py
--------------------
Fine-tunes Donut using QLoRA (4-bit quantization + LoRA adapters)
✅ Works with Kaggle T4/P100 (≤8 GB VRAM)
✅ Uses 256×256 images for efficiency
✅ Includes compatibility fixes for PEFT + Accelerate
"""

import os
import json
import torch
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ---------------- Environment setup ---------------- #
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ---------------- Load entity file ---------------- #
def load_entity_file(path):
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


# ---------------- Build JSON dataset ---------------- #
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
        gt_str = "<s_receipt>" + json.dumps(gt_json, ensure_ascii=False) + "</s>"
        donut_samples.append({
            "image": os.path.join(img_dir, fname),
            "ground_truth": gt_str
        })

    with open(output_json, "w", encoding="utf-8") as f:
        for s in donut_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(donut_samples)} samples to {output_json}")
    return output_json


# ---------------- Load dataset in memory ---------------- #
def load_json_dataset(train_json, val_json):
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(val_json, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })


# ---------------- Preprocess function ---------------- #
def make_preprocess_fn(processor):
    eos_id = processor.tokenizer.eos_token_id

    def _fn(examples):
        pixel_values, labels = [], []
        for image_path, gt in zip(examples["image"], examples["ground_truth"]):
            image = Image.open(image_path).convert("RGB").resize((256, 256))
            pv = processor(image, return_tensors="pt").pixel_values.squeeze(0)

            ids = processor.tokenizer(
                gt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.squeeze(0)
            if eos_id is not None:
                ids = torch.cat([ids, torch.tensor([eos_id], dtype=torch.long)])
            pixel_values.append(pv)
            labels.append(ids)

        return {
            "pixel_values": [torch.as_tensor(p) for p in pixel_values],
            "labels": [torch.as_tensor(l, dtype=torch.long) for l in labels],
        }

    return _fn


# ---------------- Collator ---------------- #
def donut_collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    max_len = max(x["labels"].size(0) for x in batch)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        labels[i, : x["labels"].size(0)] = x["labels"]
    return {"pixel_values": pixel_values, "labels": labels}


# ---------------- QLoRA training ---------------- #
def main():
    data_root = "/kaggle/input/receipt-dataset"
    model_id = "naver-clova-ix/donut-base"  # or Bennet1996/donut-small
    out_dir = "/kaggle/temp/outputs_donut"
    os.makedirs(out_dir, exist_ok=True)

    # 1️⃣ Build JSON
    train_json = build_donut_json(data_root, "train", os.path.join(out_dir, "train.json"))
    val_json = build_donut_json(data_root, "test", os.path.join(out_dir, "val.json"))

    # 2️⃣ Load dataset
    dataset = load_json_dataset(train_json, val_json)

    # 3️⃣ Processor + 4-bit model
    processor = DonutProcessor.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 4️⃣ Attach LoRA adapters
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # safer subset
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 5️⃣ Tokenizer & model setup
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = (
        processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id
    )
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = 128

    # 6️⃣ Preprocess
    preprocess_fn = make_preprocess_fn(processor)
    dataset = dataset.map(preprocess_fn, batched=True)
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    # 7️⃣ Memory optimization
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 8️⃣ Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=8,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=os.path.join(out_dir, "logs"),
        save_total_limit=2,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.tokenizer,
        data_collator=donut_collate_fn,
    )

    print("🚀 Starting Donut fine-tuning with QLoRA (Kaggle-optimized)...")
    trainer.train()
    print("✅ Training complete!")

    # 🔟 Save only adapters + processor
    final_dir = os.path.join(out_dir, "final_model_lora")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"🏁 Saved QLoRA fine-tuned Donut adapters to {final_dir}")


if __name__ == "__main__":
    main()
