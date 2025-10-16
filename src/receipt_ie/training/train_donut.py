"""
train_donut_lora_final_save_adapter.py
--------------------------------------
Stable LoRA fine-tuning for Donut (Bennet1996/donut-small) on Kaggle GPUs.

Key improvements:
- Uses LoRA (CAUSAL_LM) on decoder only.
- Encoder frozen for VRAM efficiency.
- Properly saves adapter (adapter_config.json + adapter_model.bin).
- Dataset built from /train/img + /train/entities structure.
"""

import os
import json
import torch
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import types

# -------------------- Environment --------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# -------------------- Utilities ----------------------
def load_entity_file(path):
    """Reads a JSON or key:value TXT file into a dict."""
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


def build_donut_json(data_root, split="train", output_json="train.json"):
    """Builds Donut JSON lines from image + entity directories."""
    img_dir = os.path.join(data_root, split, "img")
    ent_dir = os.path.join(data_root, split, "entities")

    donut_samples = []
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

        entities = load_entity_file(ent_path)
        gt_json = {
            "company": entities.get("company", ""),
            "date": entities.get("date", ""),
            "address": entities.get("address", ""),
            "total": entities.get("total", ""),
        }
        gt_str = "<s_receipt>" + json.dumps(gt_json, ensure_ascii=False) + "</s>"
        donut_samples.append({"image": os.path.join(img_dir, fname), "ground_truth": gt_str})

    with open(output_json, "w", encoding="utf-8") as f:
        for s in donut_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(donut_samples)} samples → {output_json}")
    return output_json


def load_json_dataset(train_json, val_json):
    """Loads prebuilt JSON files into HuggingFace DatasetDict."""
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(val_json, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]
    return DatasetDict({"train": Dataset.from_list(train_data), "validation": Dataset.from_list(val_data)})


def make_preprocess_fn(processor, size=224):
    eos_id = processor.tokenizer.eos_token_id

    def _fn(examples):
        pixel_values, labels = [], []
        for image_path, gt in zip(examples["image"], examples["ground_truth"]):
            image = Image.open(image_path).convert("RGB").resize((size, size))
            pv = processor(image, return_tensors="pt").pixel_values.squeeze(0)
            ids = processor.tokenizer(
                gt, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=256
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


def donut_collate_fn(batch):
    pixel_values = torch.stack([torch.as_tensor(b["pixel_values"]) for b in batch])
    lengths = [x["labels"].size(0) for x in batch]
    max_len = max(lengths)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        seq = x["labels"]
        labels[i, : seq.size(0)] = seq
    return {"pixel_values": pixel_values, "labels": labels}


# -------------------- Custom Trainer ------------------
class DonutTrainer(Seq2SeqTrainer):
    """Ensure only pixel_values + labels are passed to model."""
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(pixel_values=inputs["pixel_values"], labels=inputs["labels"])
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# -------------------- Main ---------------------------
def main():
    data_root = "/kaggle/input/receipt-dataset"
    model_id = "Bennet1996/donut-small"
    out_dir = "/kaggle/temp/outputs_donut"
    os.makedirs(out_dir, exist_ok=True)

    # 1️⃣ Build dataset
    train_json = build_donut_json(data_root, "train", os.path.join(out_dir, "train.json"))
    val_json = build_donut_json(data_root, "test", os.path.join(out_dir, "val.json"))
    dataset = load_json_dataset(train_json, val_json)

    # 2️⃣ Load processor + model
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Ensure tokenizer specials are included
    specials = {"<s_receipt>", "</s>"}
    if not set(processor.tokenizer.additional_special_tokens or []).issuperset(specials):
        processor.tokenizer.add_special_tokens({"additional_special_tokens": list(specials)})
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # 3️⃣ Config setup
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = (
        processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id
    )
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = 256
    model.config.use_cache = False
    model.decoder.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 4️⃣ LoRA on decoder (CAUSAL_LM)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # ✅ correct for MBartForCausalLM
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model.decoder = get_peft_model(model.decoder, lora_cfg)
    model.decoder.print_trainable_parameters()

    # Freeze encoder for stability
    for p in model.encoder.parameters():
        p.requires_grad = False

    # 5️⃣ Preprocess dataset
    preprocess = make_preprocess_fn(processor, size=224)
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    # 6️⃣ Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=25,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
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
    )

    trainer = DonutTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=donut_collate_fn,
    )

    print("🚀 Fine-tuning Donut with LoRA (CAUSAL_LM task)...")
    trainer.train()
    print("✅ Training complete!")

    # 7️⃣ Save model + adapter properly
    final_dir = os.path.join(out_dir, "final_model_lora")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    # --- NEW: Save LoRA adapter correctly ---
    from peft import PeftModel
    peft_save_dir = os.path.join(out_dir, "final_lora_adapter")
    os.makedirs(peft_save_dir, exist_ok=True)

    if hasattr(model.decoder, "save_pretrained"):
        model.decoder.save_pretrained(peft_save_dir)
    else:
        model.save_pretrained(peft_save_dir)

    processor.save_pretrained(peft_save_dir)
    print(f"✅ Saved LoRA adapter to {peft_save_dir}")
    print(f"🏁 All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
