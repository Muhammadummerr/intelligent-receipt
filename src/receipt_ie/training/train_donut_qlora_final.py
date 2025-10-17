"""
train_donut_lora_final_fixed.py
--------------------------------
LoRA fine-tuning for Donut (Bennet1996/donut-small) on structured receipt extraction.

✅ Fixes & improvements:
 - Handles both single & batched samples in transform().
 - Adds real <pad> token (avoids "attention mask not set" warning).
 - Faster eval: smaller validation batches + generation_config.
 - Robust entity handling (company, date, address, total).
 - CER metric for structured text accuracy.
 - LoRA applied only to decoder; encoder frozen.

Expected dataset structure:
root/
  train/
    img/*.jpg|png
    entities/*.json|txt  # each with {company,date,address,total}
  test/
    img/*.jpg|png
    entities/*.json|txt
"""

# -------------------- Environment Guards --------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_DISABLE_CACHE"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# -------------------- Imports --------------------
import json, gc, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, List
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


# -------------------- Utilities --------------------
def load_entity_file(path: str) -> Dict[str, Any]:
    """Load JSON or 'key: value' text entity file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            try:
                return json.loads(txt)
            except Exception:
                ent = {}
                for line in txt.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        ent[k.strip().lower()] = v.strip()
                return ent
    except Exception:
        return {}


def build_donut_json(data_root: str, split: str, out_json: str) -> str:
    """Create JSONL manifest for Donut training."""
    img_dir = os.path.join(data_root, split, "img")
    ent_dir = os.path.join(data_root, split, "entities")
    samples = []
    for fname in tqdm(sorted(os.listdir(img_dir)), desc=f"scan-{split}"):
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

        ent = load_entity_file(ent_path)
        gt = {
            "company": str(ent.get("company", "")).strip(),
            "date": str(ent.get("date", "")).strip(),  # keep raw format
            "address": str(ent.get("address", "")).strip(),
            "total": str(ent.get("total", "")).strip(),
        }

        s = "<s_receipt>" + json.dumps(gt, ensure_ascii=False) + "</s>"
        samples.append({"image": os.path.join(img_dir, fname), "ground_truth": s})

    with open(out_json, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(samples)} samples → {out_json}")
    return out_json


def donut_collate_fn(batch):
    """Pads labels and creates attention mask."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    lengths = [b["labels"].size(0) for b in batch]
    max_len = max(lengths)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["labels"].size(0)
        labels[i, :L] = b["labels"]
        attn[i, :L] = 1
    return {"pixel_values": pixel_values, "labels": labels, "labels_attention_mask": attn}


def levenshtein(a: str, b: str) -> int:
    """Simple edit distance (no dependencies)."""
    if a == b: return 0
    if len(a) == 0: return len(b)
    if len(b) == 0: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# -------------------- Main --------------------
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = "/kaggle/input/receipt-dataset"
    out_dir = "/kaggle/temp/outputs_donut"
    base_id = "Bennet1996/donut-small"
    os.makedirs(out_dir, exist_ok=True)

    # === Build JSON manifests ===
    train_json = build_donut_json(data_root, "train", os.path.join(out_dir, "train.json"))
    val_json = build_donut_json(data_root, "test", os.path.join(out_dir, "val.json"))
    dataset = load_dataset("json", data_files={"train": train_json, "validation": val_json})

    # === Processor + model ===
    processor = DonutProcessor.from_pretrained(base_id)
    model = VisionEncoderDecoderModel.from_pretrained(base_id)

    # --- Add task tokens and proper PAD token ---
    specials = ["<s_receipt>", "</s>"]
    added = processor.tokenizer.add_tokens(specials)
    if added > 0:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        print(f"✅ Added {added} special tokens for <s_receipt> task.")

    # Fix PAD/EOS confusion
    if processor.tokenizer.pad_token is None or (
        processor.tokenizer.pad_token_id == processor.tokenizer.eos_token_id
    ):
        processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    processor.tokenizer.pad_token = processor.tokenizer.pad_token or "<pad>"

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_receipt>")
    model.config.max_length = 256
    model.config.use_cache = False
    model.decoder.config.use_cache = False

    # --- GenerationConfig for cleaner eval ---
    model.generation_config = GenerationConfig(
        max_length=128,
        num_beams=1,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<s_receipt>"),
    )

    # === LoRA on decoder only ===
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model.decoder = get_peft_model(model.decoder, lora_cfg)
    model.decoder.print_trainable_parameters()

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # === Transform ===
    IMG_SIZE = 512
    eos_id = processor.tokenizer.eos_token_id

    def transform(example):
        def process_one(img_path, gt_text):
            image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            pv = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            ids = processor.tokenizer(
                gt_text,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).input_ids.squeeze(0)
            if eos_id is not None:
                ids = torch.cat([ids, torch.tensor([eos_id])])
            return {"pixel_values": pv, "labels": ids}

        if isinstance(example["image"], list):
            outs = [process_one(img, gt) for img, gt in zip(example["image"], example["ground_truth"])]
            return {
                "pixel_values": torch.stack([x["pixel_values"] for x in outs]),
                "labels": [x["labels"] for x in outs],
            }
        else:
            return process_one(example["image"], example["ground_truth"])

    dataset = dataset.with_transform(transform)
    gc.collect()

    # === Training arguments ===
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=2,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        predict_with_generate=True,
        generation_max_length=None,  # use model.generation_config
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=8,
        logging_dir=os.path.join(out_dir, "logs"),
        save_total_limit=2,
        report_to=[],
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        weight_decay=0.01,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # === Metrics ===
    pad_id = processor.tokenizer.pad_token_id

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds_str = processor.batch_decode(preds, skip_special_tokens=True)
        labels_np = np.where(labels != -100, labels, pad_id)
        labels_str = processor.batch_decode(labels_np, skip_special_tokens=True)
        cers = [
            levenshtein(p.strip(), l.strip()) / max(1, len(l.strip()))
            for p, l in zip(preds_str, labels_str)
            if l.strip()
        ]
        return {"cer": float(np.mean(cers)) if cers else 0.0}

    # === Trainer ===
    class DonutTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(pixel_values=inputs["pixel_values"], labels=inputs["labels"])
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = DonutTrainer(
        model=model.to(device),
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=donut_collate_fn,
        compute_metrics=compute_metrics,
    )

    # === Train ===
    print("🚀 Training started...")
    trainer.train()
    print("✅ Training complete.")

    # === Save ===
    final_base = os.path.join(out_dir, "final_model_lora")
    adapter_dir = os.path.join(out_dir, "final_lora_adapter")

    model.save_pretrained(final_base)
    processor.save_pretrained(final_base)

    model.decoder.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    print(f"✅ Final model saved: {final_base}")
    print(f"✅ Adapter saved: {adapter_dir}")


if __name__ == "__main__":
    main()
