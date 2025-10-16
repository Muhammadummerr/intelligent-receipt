"""
train_donut_lora_final.py
-------------------------
Fine-tune Donut using LoRA adapters (no bitsandbytes)
✅ Works on Kaggle GPUs (T4 / P100)
✅ 256×256 image resizing
✅ In-memory dataset
✅ Gradient checkpointing + accumulation
"""

import os, json, torch
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ------------------ Load entity file ------------------ #
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
    except Exception:
        return {}


# ------------------ Build dataset JSON ------------------ #
def build_donut_json(data_root, split, out_path):
    img_dir = os.path.join(data_root, split, "img")
    ent_dir = os.path.join(data_root, split, "entities")
    samples = []
    for fname in tqdm(sorted(os.listdir(img_dir)), desc=f"{split}"):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        ent_file = next((os.path.join(ent_dir, stem + e)
                         for e in (".json", ".txt")
                         if os.path.isfile(os.path.join(ent_dir, stem + e))), None)
        if not ent_file:
            continue
        ent = load_entity_file(ent_file)
        gt = "<s_receipt>" + json.dumps({
            "company": ent.get("company", ""),
            "date": ent.get("date", ""),
            "address": ent.get("address", ""),
            "total": ent.get("total", "")
        }, ensure_ascii=False) + "</s>"
        samples.append({"image": os.path.join(img_dir, fname), "ground_truth": gt})
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(samples)} samples → {out_path}")
    return out_path


def load_json_dataset(train_json, val_json):
    def load_file(p): return [json.loads(line) for line in open(p, encoding="utf-8")]
    return DatasetDict({
        "train": Dataset.from_list(load_file(train_json)),
        "validation": Dataset.from_list(load_file(val_json))
    })


# ------------------ Preprocessing ------------------ #
def make_preprocess_fn(processor):
    eos = processor.tokenizer.eos_token_id

    def fn(batch):
        pix, lbl = [], []
        for img_path, gt in zip(batch["image"], batch["ground_truth"]):
            img = Image.open(img_path).convert("RGB").resize((256, 256))
            pv = processor(img, return_tensors="pt").pixel_values.squeeze(0)
            ids = processor.tokenizer(gt, add_special_tokens=False,
                                      return_tensors="pt").input_ids.squeeze(0)
            if eos is not None:
                ids = torch.cat([ids, torch.tensor([eos], dtype=torch.long)])
            pix.append(pv)
            lbl.append(ids)
        return {"pixel_values": pix, "labels": lbl}
    return fn


# ------------------ Collator ------------------ #
def donut_collate(batch):
    pix = torch.stack([torch.as_tensor(b["pixel_values"]) for b in batch])
    max_len = max(len(b["labels"]) for b in batch)
    lbl = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        seq = torch.as_tensor(b["labels"], dtype=torch.long)
        lbl[i, :len(seq)] = seq
    return {"pixel_values": pix, "labels": lbl}


# ------------------ Main Training ------------------ #
def main():
    data_root = "/kaggle/input/receipt-dataset"
    model_id = "Bennet1996/donut-small"
    out_dir = "/kaggle/temp/outputs_donut"
    os.makedirs(out_dir, exist_ok=True)

    train_json = build_donut_json(data_root, "train", f"{out_dir}/train.json")
    val_json   = build_donut_json(data_root, "test",  f"{out_dir}/val.json")
    ds = load_json_dataset(train_json, val_json)

    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Attach LoRA adapters
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = (
        processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id)
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = 128

    ds = ds.map(make_preprocess_fn(processor), batched=True)
    ds.set_format(type="torch", columns=["pixel_values", "labels"])

    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=10,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=f"{out_dir}/logs",
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=processor.tokenizer,
        data_collator=donut_collate,
    )

    print("🚀 Fine-tuning Donut with LoRA (no bitsandbytes)…")
    trainer.train()
    print("✅ Training complete!")

    final = f"{out_dir}/final_model_lora"
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"🏁 Saved LoRA-fine-tuned model to {final}")


if __name__ == "__main__":
    main()
