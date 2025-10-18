# -*- coding: utf-8 -*-
"""
Train a ViT classifier to detect watermarked vs clean receipts.

Expected dataset layout (from your builder):
dataset_root/
  train/
    clean/
    watermarked/
  val/
    clean/
    watermarked/
  test/
    clean/
    watermarked/

Run (Kaggle):
!python -m src.receipt_ie.training.train_vit_watermark \
  --data_root /kaggle/working/receipt-watermark-advanced \
  --out_dir /kaggle/working/wm_vit_out \
  --epochs 10 --batch 32 --lr 5e-5
"""

import os
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ----------------------
# Torch dataset wrappers
# ----------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def scan_split(split_dir: str) -> Tuple[list, list]:
    """
    Returns (paths, labels) for a split dir that contains:
      split_dir/clean/*.jpg
      split_dir/watermarked/*.jpg
    Label map: clean->0, watermarked->1
    """
    paths, labels = [], []
    for label_name, label_id in [("clean", 0), ("watermarked", 1)]:
        d = os.path.join(split_dir, label_name)
        if not os.path.isdir(d):
            # tolerate missing; just skip
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(d, fn))
                labels.append(label_id)
    return paths, labels


class ReceiptWMDataset(Dataset):
    def __init__(self, paths, labels, processor: ViTImageProcessor, augment: bool = False):
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.augment = augment

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = int(self.labels[idx])

        # Open in RGB
        with Image.open(p) as im:
            im = im.convert("RGB")

        # Light but helpful augmentations that preserve receipt structure
        if self.augment:
            # random slight rotation / perspective / brightness jitter
            # (keep it PIL + processor-friendly)
            if np.random.rand() < 0.30:
                angle = np.random.uniform(-2.0, 2.0)
                im = im.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
            if np.random.rand() < 0.25:
                # brightness jitter
                from PIL import ImageEnhance
                im = ImageEnhance.Brightness(im).enhance(np.random.uniform(0.9, 1.1))
            if np.random.rand() < 0.20:
                # contrast jitter
                from PIL import ImageEnhance
                im = ImageEnhance.Contrast(im).enhance(np.random.uniform(0.9, 1.1))
            if np.random.rand() < 0.20:
                # tiny Gaussian noise
                arr = np.array(im).astype(np.float32)
                arr += np.random.normal(0, 3.0, arr.shape)  # small noise
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                im = Image.fromarray(arr)

        pixel_values = self.processor(images=im, return_tensors="pt")["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "labels": torch.tensor(y, dtype=torch.long)}


# ----------------------
# Metrics
# ----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits.argmax(-1)).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ----------------------
# Custom Trainer for weighted CE
# ----------------------
class WeightedCETrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(pixel_values=inputs["pixel_values"], labels=None)
        logits = outputs.logits  # (B, 2)

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ----------------------
# Inference helper
# ----------------------
def quick_predict(model_dir: str, image_path: str) -> Dict[str, float]:
    processor = ViTImageProcessor.from_pretrained(model_dir)
    model = ViTForImageClassification.from_pretrained(model_dir)
    model.eval()
    with Image.open(image_path) as im:
        im = im.convert("RGB")
    with torch.no_grad():
        pv = processor(images=im, return_tensors="pt")["pixel_values"]
        logits = model(pixel_values=pv).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    # id2label stored in config
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}


# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Dataset root that contains train/ val/ test/")
    ap.add_argument("--out_dir", required=True, help="Where to save checkpoints/metrics")
    ap.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Scan splits
    train_paths, train_labels = scan_split(os.path.join(args.data_root, "train"))
    # val_paths, val_labels     = scan_split(os.path.join(args.data_root, "val"))
    val_paths, val_labels    = scan_split(os.path.join(args.data_root, "test"))

    assert train_paths, "No training images found."
    assert val_paths, "No validation images found."

    print(f"Found: train={len(train_paths)} val={len(val_paths)}")

    # Processor & model
    processor = ViTImageProcessor.from_pretrained(args.model_name)
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "clean", 1: "watermarked"},
        label2id={"clean": 0, "watermarked": 1},
        ignore_mismatched_sizes=True,   # allows head resize
    )

    # Datasets
    train_ds = ReceiptWMDataset(train_paths, train_labels, processor, augment=True)
    val_ds   = ReceiptWMDataset(val_paths, val_labels, processor, augment=False)
    # test_ds  = ReceiptWMDataset(test_paths, test_labels, processor, augment=False) if test_paths else None

    # Class weights (handle imbalance)
    cls_counts = np.bincount(train_labels, minlength=2)
    majority = max(cls_counts.max(), 1)
    weights = majority / np.clip(cls_counts, 1, None)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"Class counts train: {cls_counts.tolist()}  -> class_weights: {weights}")

    # Training args
    targs = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedCETrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,  # for nice logging only
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("🚀 Training start")
    trainer.train()
    print("✅ Training complete")

    # Save best model
    trainer.save_model(args.out_dir)
    processor.save_pretrained(args.out_dir)

    # Quick evaluation on val + optional test
    def eval_split(name, ds, y_true):
        preds = trainer.predict(ds)
        y_pred = preds.predictions.argmax(-1)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        print(f"\n[{name}]  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
        print("Confusion matrix [rows=true clean, watermarked]:\n", cm)

    eval_split("VAL", val_ds, val_labels)
    # if test_ds:
    #     eval_split("TEST", test_ds, test_labels)

    # Demo prediction print (first val sample)
    demo_path = val_paths[0]
    probs = quick_predict(args.out_dir, demo_path)
    print(f"\n🔎 Demo prediction on {os.path.basename(demo_path)} -> {probs}")


if __name__ == "__main__":
    main()
