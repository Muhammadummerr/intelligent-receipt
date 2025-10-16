"""
train_donut_lora_final_dp_safe.py
---------------------------------
Stable LoRA fine-tuning for Donut on Kaggle GPUs (no bitsandbytes).

Key points:
- LoRA on decoder (language side), encoder frozen
- Bypass VisionEncoderDecoder.forward in training (call decoder directly)
- DataParallel-safe: unwrap model inside compute_loss
- Kaggle-friendly memory footprint
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
    """Builds Donut JSONL from image + entity directories."""
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
            ids = processor.tokenizer(gt, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
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
    """
    DataParallel-safe custom training step:
      - unwrap model before accessing get_encoder()/decoder
      - run encoder explicitly, then call decoder with input_ids (not decoder_input_ids)
    """

    @staticmethod
    def _unwrap(m):
        return m.module if hasattr(m, "module") else m

    def _shift_right(self, labels, pad_token_id, start_token_id):
        dec = labels.clone()
        dec[dec == -100] = pad_token_id
        shifted = dec.new_full(dec.shape, pad_token_id)
        shifted[:, 1:] = dec[:, :-1]
        shifted[:, 0] = start_token_id
        return shifted

    def compute_loss(self, model, inputs, return_outputs=False):
        base = self._unwrap(model)

        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        # 1) encode image
        encoder_outputs = base.get_encoder()(pixel_values=pixel_values, return_dict=True)

        # 2) prepare decoder inputs (teacher forcing)
        pad_id = base.config.pad_token_id
        start_id = base.config.decoder_start_token_id
        decoder_input_ids = self._shift_right(labels, pad_id, start_id)

        # 3) forward through decoder with correct arg names
        dec = base.decoder  # MBartForCausalLM (may be PEFT-wrapped)
        outputs = dec(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# -------------------- Main ---------------------------
def main():
    data_root = "/kaggle/input/receipt-dataset"
    model_id = "Bennet1996/donut-small"  # Lightweight Donut
    out_dir = "/kaggle/temp/outputs_donut"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build dataset JSONL
    train_json = build_donut_json(data_root, "train", os.path.join(out_dir, "train.json"))
    val_json = build_donut_json(data_root, "test", os.path.join(out_dir, "val.json"))
    dataset = load_json_dataset(train_json, val_json)

    # 2) Load processor + model
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # --- embedding routing for PEFT safety (no-op for our training path, but harmless) ---
    def _get_input_embeddings(self):
        return self.decoder.get_input_embeddings()
    def _set_input_embeddings(self, new_emb):
        return self.decoder.set_input_embeddings(new_emb)
    model.get_input_embeddings = types.MethodType(_get_input_embeddings, model)
    model.set_input_embeddings = types.MethodType(_set_input_embeddings, model)
    # --------------------------------------------------------------------------------------

    # 3) Tokenizer + config
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = 128
    model.config.use_cache = False
    model.decoder.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 4) LoRA on decoder (full MBartForCausalLM is fine since we call it directly)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model.decoder = get_peft_model(model.decoder, lora_cfg)
    try:
        model.decoder.print_trainable_parameters()
    except Exception:
        trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params (decoder): {trainable} / {total} ({100.0 * trainable / total:.4f}%)")

    # ✅ Freeze encoder (vision side)
    for p in model.encoder.parameters():
        p.requires_grad = False

    # 5) Preprocess
    preprocess = make_preprocess_fn(processor, size=224)
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    # 6) Training args — keep generate off to avoid the buggy VE forward path
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=10,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        predict_with_generate=False,  # important
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

    print("🚀 Fine-tuning Donut (encoder frozen) with LoRA on decoder…")
    trainer.train()
    print("✅ Training complete!")

    final_dir = os.path.join(out_dir, "final_model_lora")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"🏁 Saved LoRA adapters + processor to {final_dir}")


if __name__ == "__main__":
    main()
