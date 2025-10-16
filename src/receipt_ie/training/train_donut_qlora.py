"""
train_donut_lora_kaggle.py
--------------------------
LoRA fine-tuning for Donut on Kaggle GPUs (no bitsandbytes).
- Works on T4/P100 (<=8GB VRAM)
- Image resize 224x224
- In-memory dataset (no Arrow cache)
- Custom Trainer that never passes `input_ids` to the vision encoder
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

# -------------------- Environment --------------------
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# -------------------- Utilities ----------------------
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


def build_donut_json(data_root, split="train", output_json="train.json"):
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


def make_preprocess_fn(processor):
    eos_id = processor.tokenizer.eos_token_id

    def _fn(examples):
        pixel_values, labels = [], []
        for image_path, gt in zip(examples["image"], examples["ground_truth"]):
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224))  # small, Kaggle-safe

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
    """Force the model to receive only pixel_values + labels.
    Prevents accidental passing of `input_ids` to the vision encoder.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(pixel_values=inputs["pixel_values"], labels=inputs["labels"])
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self, model, inputs, prediction_loss_only: bool, ignore_keys=None
    ):
        with torch.no_grad():
            outputs = model(pixel_values=inputs["pixel_values"], labels=inputs["labels"])
            loss = outputs.loss.detach()

            if prediction_loss_only or not self.args.predict_with_generate:
                return (loss, None, None)

            gen = model.generate(
                pixel_values=inputs["pixel_values"],
                max_length=self.args.generation_max_length,
            )
            return (loss, gen, inputs["labels"])


# -------------------- Main ---------------------------
def main():
    data_root = "/kaggle/input/receipt-dataset"
    # Use the small model by default (fits T4/P100 comfortably)
    model_id = "Bennet1996/donut-small"
    out_dir = "/kaggle/temp/outputs_donut"
    os.makedirs(out_dir, exist_ok=True)

    # Build dataset jsons
    train_json = build_donut_json(data_root, "train", os.path.join(out_dir, "train.json"))
    val_json = build_donut_json(data_root, "test", os.path.join(out_dir, "val.json"))
    dataset = load_json_dataset(train_json, val_json)

    # Processor & base model
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    # --- PATCH: prevent input_ids being sent to the vision encoder ---
    def patched_forward(self, pixel_values=None, labels=None, **kwargs):
        """Filters out invalid kwargs before passing to encoder."""
        # Filter any 'input_ids' or text-only arguments
        encoder_kwargs = {
            k: v for k, v in kwargs.items() if k in ["pixel_values", "attention_mask", "output_attentions", "output_hidden_states", "return_dict"]
        }

        encoder_outputs = self.encoder(pixel_values=pixel_values, **encoder_kwargs)
        encoder_hidden_states = encoder_outputs[0]

        # Decoder forward pass (teacher forcing if labels provided)
        decoder_inputs = {
            "encoder_hidden_states": encoder_hidden_states,
            "labels": labels,
            "return_dict": True,
        }
        return self.decoder(**decoder_inputs)

    # Monkey patch the forward method
    import types
    model.forward = types.MethodType(patched_forward, model)
    print("✅ Patched model.forward() to ignore 'input_ids' for Donut encoder.")


    # Tokenizer/model config
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = 128

    # Attach LoRA adapters (decoder-only modules)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Preprocess
    preprocess = make_preprocess_fn(processor)
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    # Memory helpers
    model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Training args
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=10,
        learning_rate=2e-4,                 # LoRA: you can use higher LR
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,        # <- important for vision enc/dec
        report_to=[],                       # no W&B
    )

    trainer = DonutTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        # Do NOT pass tokenizer here; we’re not using text input_ids
        data_collator=donut_collate_fn,
    )

    print("🚀 Fine-tuning Donut with LoRA (Kaggle-safe)…")
    trainer.train()
    print("✅ Training complete!")

    final_dir = os.path.join(out_dir, "final_model_lora")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"🏁 Saved LoRA adapters + processor to {final_dir}")


if __name__ == "__main__":
    main()
