"""
train_layoutlmv3.py
-------------------
Fine-tunes LayoutLMv3 for document information extraction on receipt OCR data.

Config-driven training pipeline:
- Loads hyperparameters & paths from configs/default.yaml
- Uses Dataset + Collator from src/receipt_ie/data
- Saves best model and processor for inference
"""

import os
import yaml
import torch
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from src.receipt_ie.data.dataset_layoutlmv3 import ReceiptLayoutLMv3Dataset
from src.receipt_ie.data.collate import identity_collate
from src.receipt_ie.data.align import LABELS, LABEL2ID


# -----------------------------------------------------
# Load configuration
# -----------------------------------------------------
def load_config(cfg_path="configs/default.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------
# Training Entry Point
# -----------------------------------------------------
def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    log_cfg = cfg["logging"]

    output_dir = os.path.join(log_cfg.get("out_dir", "./outputs"), "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # --- Initialize processor ---
    model_name = "microsoft/layoutlmv3-base"
    processor = LayoutLMv3Processor.from_pretrained(model_name,apply_ocr=False)

    print("📂 Loading training & validation datasets...")
    train_ds = ReceiptLayoutLMv3Dataset(
        img_dir=data_cfg["train_img_dir"],
        box_dir=data_cfg["train_box_dir"],
        ent_dir=data_cfg["train_entities_dir"],
        processor=processor,
        max_seq_len=model_cfg["max_seq_len"],
    )
    val_ds = ReceiptLayoutLMv3Dataset(
        img_dir=data_cfg["train_img_dir"],
        box_dir=data_cfg["train_box_dir"],
        ent_dir=data_cfg["train_entities_dir"],
        processor=processor,
        max_seq_len=model_cfg["max_seq_len"],
    )

    # --- Initialize model ---
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL2ID,
    )

    # --- Data collator ---
    data_collator = identity_collate 

    # --- Training arguments ---
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1 if model_cfg.get("save_best_only", True) else 3,
        learning_rate=model_cfg["lr"],
        weight_decay=model_cfg.get("weight_decay", 0.01),
        per_device_train_batch_size=model_cfg["batch_size"],
        per_device_eval_batch_size=model_cfg["batch_size"],
        num_train_epochs=model_cfg["epochs"],
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(log_cfg.get("out_dir", "./outputs"), "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    # --- Trainer setup ---
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
    )

    # --- Training ---
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete.")

    # --- Save best model + processor ---
    final_dir = os.path.join(log_cfg.get("out_dir", "./outputs"), "final")
    os.makedirs(final_dir, exist_ok=True)

    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print(f"🏁 Model and processor saved to: {final_dir}")


if __name__ == "__main__":
    main()
