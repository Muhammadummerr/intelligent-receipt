"""
train_donut_lora_kaggle_safe.py
--------------------------------
Kaggle-safe LoRA fine-tuning for Donut (Bennet1996/donut-small).

Memory tactics:
- Force single GPU (no NCCL / DP).
- Image size 192x192.
- On-the-fly transforms via set_transform (no full-tensor map in RAM).
- No gradient checkpointing (reduces host RAM).
- bf16 on T4 (stable).
- Tiny batch (1) + small grad_accum (2).

LoRA:
- TaskType.CAUSAL_LM on decoder (MBartForCausalLM).
- Saves adapter files for clean inference (adapter_config.json + adapter_model.bin).
"""

# -------------------- Hard memory/parallel guards (must be first) --------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Reduce background/thread usage
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_DISABLE_CACHE"] = "0"   # we WANT disk cache
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import json
import gc
from typing import Dict, Any

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


# -------------------- Utility: build JSONL (same structure as before) --------------------
def load_entity_file(path: str) -> Dict[str, Any]:
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
    import os
    from tqdm import tqdm

    img_dir = os.path.join(data_root, split, "img")
    ent_dir = os.path.join(data_root, split, "entities")
    samples = []
    for fname in tqdm(sorted(os.listdir(img_dir)), desc=split):
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


# -------------------- Collate (kept simple & lightweight) --------------------
def donut_collate_fn(batch):
    # batch elements already have tensors; just stack
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    lengths = [b["labels"].size(0) for b in batch]
    max_len = max(lengths)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["labels"].size(0)
        labels[i, :L] = b["labels"]
    return {"pixel_values": pixel_values, "labels": labels}


# -------------------- Main --------------------
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = "/kaggle/input/receipt-dataset"
    out_dir = "/kaggle/temp/outputs_donut"
    base_id = "Bennet1996/donut-small"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build JSONL manifest
    train_json = build_donut_json(data_root, "train", os.path.join(out_dir, "train.json"))
    val_json   = build_donut_json(data_root, "test",  os.path.join(out_dir, "val.json"))

    # 2) Load datasets from disk JSON (keeps RAM low)
    dataset = load_dataset(
        "json",
        data_files={"train": train_json, "validation": val_json},
        cache_dir="/kaggle/temp/hf_cache"
    )

    # 3) Processor + model
    processor = DonutProcessor.from_pretrained(base_id)
    model = VisionEncoderDecoderModel.from_pretrained(base_id)

    # Ensure our task-specific specials exist
    specials = {"<s_receipt>", "</s>"}
    if not set(processor.tokenizer.additional_special_tokens or []).issuperset(specials):
        processor.tokenizer.add_special_tokens({"additional_special_tokens": list(specials)})
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Config & memory: SMALLER image, longer text cap, NO checkpointing
    IMG_SIZE = 192
    model.config.max_length = 256
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.use_cache = False
    model.decoder.config.use_cache = False
    # IMPORTANT: disable checkpointing to save host RAM
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    # 4) LoRA on decoder (CAUSAL_LM) — light footprint
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model.decoder = get_peft_model(model.decoder, lora_cfg)
    model.decoder.print_trainable_parameters()

    # Freeze encoder (vision side)
    for p in model.encoder.parameters():
        p.requires_grad = False

    # 5) Lazy/on-the-fly transform to avoid storing big tensors in RAM
    eos_id = processor.tokenizer.eos_token_id

    def transform(example):
    # Handle both single and batched examples
        def process_one(img_path, gt_text):
            image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            pv = processor(image, return_tensors="pt").pixel_values.squeeze(0)
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

        if isinstance(example["image"], list):  # batched
            pixel_values, labels = [], []
            for img, gt in zip(example["image"], example["ground_truth"]):
                pv, ids = process_one(img, gt)
                pixel_values.append(pv)
                labels.append(ids)
            return {"pixel_values": pixel_values, "labels": labels}
        else:  # single
            pv, ids = process_one(example["image"], example["ground_truth"])
            return {"pixel_values": pv, "labels": ids}


    dataset["train"] = dataset["train"].with_transform(transform)
    dataset["validation"] = dataset["validation"].with_transform(transform)

    # free some python objects
    gc.collect()

    # 6) Training args — bf16, tiny batch, small grad_accum
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=20,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        predict_with_generate=False,   # avoid VE.generate() during eval
        bf16=True, fp16=False,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,   # we depend on custom collate & transform
        report_to=[],
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,      # light workers to avoid RAM spikes
        # max_split_size_mb=32,
    )

    # 7) Trainer (simple: call model.forward)
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
    )

    print("🚀 Training (memory-optimized)…")
    trainer.train()
    print("✅ Training done.")

    # 8) Save base wrapper + processor
    final_base = os.path.join(out_dir, "final_model_lora")
    os.makedirs(final_base, exist_ok=True)
    model.save_pretrained(final_base)
    processor.save_pretrained(final_base)

    # 9) Save the LoRA adapter cleanly (for inference with PeftModel)
    adapter_dir = os.path.join(out_dir, "final_lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    if hasattr(model.decoder, "save_pretrained"):
        model.decoder.save_pretrained(adapter_dir)
    else:
        model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    print(f"✅ Saved adapter to: {adapter_dir}")
    print(f"🏁 All artifacts in: {out_dir}")


if __name__ == "__main__":
    # extra guard: pin device (avoids DP surprises)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    main()
