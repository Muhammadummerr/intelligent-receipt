"""
train_donut_lora_final.py
--------------------------------
Kaggle-safe LoRA fine-tuning for Donut (Bennet1996/donut-small).

Key improvements vs. previous version:
- Use 512x512 images (small text becomes readable).
- Use processor(images=...) for correct normalization.
- Predict with generate + CER metric during validation.
- Saner hyperparams to avoid overfit on small receipt sets.
- Keep LoRA on decoder only; freeze vision encoder.

I/O expectation (dataset root):
root/
  train/
    img/*.jpg|png
    entities/*.txt|json    # {company,date,address,total}
  test/
    img/*.jpg|png
    entities/*.txt|json
"""

# -------------------- Hard memory/parallel guards (must be first) --------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_DISABLE_CACHE"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import json
import gc
import numpy as np
from typing import Dict, Any, List

import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


# -------------------- Utilities --------------------
def load_entity_file(path: str) -> Dict[str, Any]:
    """Tolerant loader for .json or plain 'k: v' .txt"""
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
    """Create JSONL manifest: {image: path, ground_truth: "<s_receipt>{...}</s>"}"""
    import os
    from tqdm import tqdm

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
            "company": ent.get("company", ""),
            "date": ent.get("date", ""),
            "address": ent.get("address", ""),
            "total": ent.get("total", ""),
        }
        s = "<s_receipt>" + json.dumps(gt, ensure_ascii=False) + "</s>"
        samples.append({"image": os.path.join(img_dir, fname), "ground_truth": s})

    with open(out_json, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(samples)} samples → {out_json}")
    return out_json


def donut_collate_fn(batch):
    """Pads labels to max length; pixel_values already tensors."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    lengths = [b["labels"].size(0) for b in batch]
    max_len = max(lengths)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["labels"].size(0)
        labels[i, :L] = b["labels"]
    return {"pixel_values": pixel_values, "labels": labels}


def levenshtein(a: str, b: str) -> int:
    """Simple DP edit distance to avoid external deps."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, replace))
        prev = cur
    return prev[-1]


# -------------------- Main --------------------
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths (adjust for your Kaggle dataset mount) ---
    data_root = "/kaggle/input/receipt-dataset"   # <-- change if needed
    out_dir = "/kaggle/temp/outputs_donut"
    base_id = "Bennet1996/donut-small"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build manifests
    train_json = build_donut_json(data_root, "train", os.path.join(out_dir, "train.json"))
    val_json   = build_donut_json(data_root, "test",  os.path.join(out_dir, "val.json"))

    # 2) Load datasets
    dataset = load_dataset(
        "json",
        data_files={"train": train_json, "validation": val_json},
        cache_dir="/kaggle/temp/hf_cache"
    )

    # 3) Processor + model
    processor = DonutProcessor.from_pretrained(base_id)
    model = VisionEncoderDecoderModel.from_pretrained(base_id)

    # Add task-specific specials
    specials = {"<s_receipt>", "</s>"}
    if not set(processor.tokenizer.additional_special_tokens or []).issuperset(specials):
        processor.tokenizer.add_special_tokens({"additional_special_tokens": list(specials)})
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Config
    model.config.max_length = 256
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = (
        processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id
    )
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.use_cache = False
    model.decoder.config.use_cache = False
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    # 4) LoRA on decoder
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model.decoder = get_peft_model(model.decoder, lora_cfg)
    model.decoder.print_trainable_parameters()

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # 5) Transform (on-the-fly)
    IMG_SIZE = 512
    eos_id = processor.tokenizer.eos_token_id

    def transform(example):
        def process_one(img_path, gt_text):
            image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            enc = processor(images=image, return_tensors="pt")   # proper normalization
            pv = enc.pixel_values.squeeze(0)

            ids = processor.tokenizer(
                gt_text,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).input_ids.squeeze(0)
            if eos_id is not None:
                ids = torch.cat([ids, torch.tensor([eos_id], dtype=torch.long)])
            return pv, ids

        if isinstance(example["image"], list):
            pixel_values, labels = [], []
            for img, gt in zip(example["image"], example["ground_truth"]):
                pv, ids = process_one(img, gt)
                pixel_values.append(pv)
                labels.append(ids)
            return {"pixel_values": pixel_values, "labels": labels}
        else:
            pv, ids = process_one(example["image"], example["ground_truth"])
            return {"pixel_values": pv, "labels": ids}

    dataset = dataset.with_transform(transform)

    gc.collect()

    # 6) Training args — tuned for small datasets
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=22,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        predict_with_generate=True,
        generation_max_length=256,
        fp16=True,  # T4 friendly
        bf16=False,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
    )

    # 7) Metrics (CER on decoded strings)
    pad_token_id = processor.tokenizer.pad_token_id

    def _decode_label_batch(y: np.ndarray) -> List[str]:
        y = np.where(y != -100, y, pad_token_id)
        return processor.batch_decode(y, skip_special_tokens=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        pred_str = processor.batch_decode(preds, skip_special_tokens=True)
        label_str = _decode_label_batch(labels)
        # CER (character error rate)
        cers = []
        for p, l in zip(pred_str, label_str):
            l = l.strip()
            p = p.strip()
            if len(l) == 0:
                continue
            dist = levenshtein(p, l)
            cers.append(dist / max(1, len(l)))
        cer = float(np.mean(cers)) if cers else 0.0
        return {"cer": cer}

    # 8) Trainer
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

    print("🚀 Training…")
    trainer.train()
    print("✅ Training done.")

    # 9) Save base wrapper + processor
    final_base = os.path.join(out_dir, "final_model_lora")
    os.makedirs(final_base, exist_ok=True)
    model.save_pretrained(final_base)
    processor.save_pretrained(final_base)

    # 10) Save LoRA adapter separately
    adapter_dir = os.path.join(out_dir, "final_lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.decoder.save_pretrained(adapter_dir)  # PEFT saves adapter_config + adapter_model
    processor.save_pretrained(adapter_dir)

    print(f"✅ Saved adapter to: {adapter_dir}")
    print(f"🏁 All artifacts in: {out_dir}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    main()
