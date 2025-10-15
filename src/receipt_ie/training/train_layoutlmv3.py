"""
train_layoutlmv3.py
-------------------
Fine-tunes LayoutLMv3 for document information extraction on receipt OCR data.

This version adds:
✅ Class-weighted loss for label imbalance (via custom Trainer)
✅ Token-label alignment sanity check
✅ F1-score metric during evaluation
✅ OCR caching + dynamic OCR (EasyOCR)
"""

import os
import yaml
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

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
# Utility: load config
# -----------------------------------------------------
def load_config(cfg_path="configs/default.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------
# Metrics
# -----------------------------------------------------
def compute_metrics(p):
    # p.predictions: (batch, seq, num_labels)
    preds = np.argmax(p.predictions, axis=-1).flatten()
    labels = p.label_ids.flatten()
    mask = labels != -100
    preds, labels = preds[mask], labels[mask]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that applies class-weighted CrossEntropyLoss.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard HF keys: input_ids, attention_mask, bbox, pixel_values, labels
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")  # (B, T, C)

        # Class-weighted CE, ignore_index=-100 for padding tokens
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            ignore_index=-100,
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------
# Training Entry Point
# -----------------------------------------------------
def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    log_cfg = cfg["logging"]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silence fork warning

    output_dir = os.path.join(log_cfg.get("out_dir", "./outputs"), "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    model_name = "microsoft/layoutlmv3-base"
    processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)

    print("📂 Loading datasets...")
    train_ds = ReceiptLayoutLMv3Dataset(
        img_dir=data_cfg["train_img_dir"],
        box_dir=data_cfg["train_box_dir"],
        ent_dir=data_cfg["train_entities_dir"],
        processor=processor,
        max_seq_len=model_cfg["max_seq_len"],
        use_easyocr=True,
        cache_dir="./ocr_cache",
    )
    val_ds = ReceiptLayoutLMv3Dataset(
        img_dir=data_cfg["train_img_dir"],
        box_dir=data_cfg["train_box_dir"],
        ent_dir=data_cfg["train_entities_dir"],
        processor=processor,
        max_seq_len=model_cfg["max_seq_len"],
        use_easyocr=True,
        cache_dir="./ocr_cache",
    )

    # -------------------------------------------------
    # 🔍 Label distribution sanity check
    # -------------------------------------------------
    print("🔍 Checking label distribution (sample of train set)...")
    label_counts = Counter()
    sample_n = min(200, len(train_ds))  # increase if fast enough
    for i in range(sample_n):
        sample = train_ds[i]
        labels = sample["labels"].numpy().tolist()
        label_counts.update([l for l in labels if l != -100])
    inv_map = {v: k for k, v in LABEL2ID.items()}
    for k, v in label_counts.most_common():
        print(f"{inv_map.get(k, k):<12}: {v}")
    total_labeled = sum(label_counts.values()) or 1
    frac_o = label_counts.get(LABEL2ID["O"], 0) / total_labeled
    if frac_o > 0.95:
        print("⚠️ Warning: >95% of tokens are 'O' — entity signal is very sparse. Consider improving alignment/labels.")

    # -------------------------------------------------
    # Model init (no monkey-patching!)
    # -------------------------------------------------
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL2ID,
    )

    # -------------------------------------------------
    # Class weights from label distribution
    # -------------------------------------------------
    class_weights = torch.ones(len(LABELS), dtype=torch.float)
    o_weight = label_counts.get(LABEL2ID["O"], 1)
    for label, idx in LABEL2ID.items():
        if label != "O":
            # inverse-frequency-ish; clamp to keep things stable
            w = o_weight / max(label_counts.get(idx, 1), 1)
            class_weights[idx] = float(np.clip(w, 1.0, 50.0))
    print("⚖️ Using class weights:", [round(x, 3) for x in class_weights.tolist()])

    # -------------------------------------------------
    # Training setup
    # -------------------------------------------------
    data_collator = identity_collate

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
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete.")

    # Save model
    final_dir = os.path.join(log_cfg.get("out_dir", "./outputs"), "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print(f"🏁 Model and processor saved to: {final_dir}")


if __name__ == "__main__":
    main()
