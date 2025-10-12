import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silence fork warning & save a bit of RAM
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")        # keep async; if debugging, set "1"

import yaml, argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from transformers import LayoutLMv3Processor, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch

from ..data.dataset_layoutlmv3 import ReceiptLayoutLMv3Dataset, label_mappings
from ..data.splits import list_stems, make_train_val, save_split, load_split
from ..data.collate import identity_collate


@dataclass
class Paths:
    data_root: str
    train_img: str
    train_box: str
    train_entities: str
    work_dir: str

def build_paths(cfg: Dict, work_dir: str) -> Paths:
    return Paths(
        data_root=cfg["data"]["root"],
        train_img=cfg["data"]["train_img_dir"],
        train_box=cfg["data"]["train_box_dir"],
        train_entities=cfg["data"]["train_entities_dir"],
        work_dir=work_dir
    )

def get_label_lists():
    L2I, I2L = label_mappings()
    id2label = {i:l for l,i in L2I.items()}
    label2id = {l:i for i,l in id2label.items()}
    return label2id, id2label

def as_hf_dataset(torch_ds: ReceiptLayoutLMv3Dataset):
    # Wrap our map-style dataset for HF Trainer
    # We'll lazy-index items; Trainer will collate tensors automatically.
    class _HF(Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            return self.base[idx]
    return _HF(torch_ds)

def compute_metrics_builder(id2label):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred  # preds: (bs, seq, n_labels), labels: (bs, seq)
        pred_ids = np.argmax(preds, axis=-1)
        mask = labels != -100
        correct = (pred_ids[mask] == labels[mask]).sum()
        total = mask.sum()
        acc = float(correct) / float(total) if total > 0 else 0.0
        return {"token_acc": acc}
    return compute_metrics


def main(args):
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    paths = build_paths(cfg, work_dir=args.out_dir)

    # list stems and split (or load existing split)
    split_dir = os.path.join(paths.work_dir, "splits")
    if args.split_dir and os.path.isdir(args.split_dir):
        train_stems, val_stems = load_split(args.split_dir)
    else:
        stems = list_stems(paths.train_img)
        train_stems, val_stems = make_train_val(stems, val_frac=args.val_frac, seed=cfg.get("seed", 42))
        save_split(train_stems, val_stems, split_dir)
        print(f"Saved split to {split_dir} -> train={len(train_stems)} val={len(val_stems)}")

    # processor & datasets
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    label2id, id2label = get_label_lists()

    train_ds = ReceiptLayoutLMv3Dataset(
    img_dir=paths.train_img,
    box_dir=paths.train_box,
    ent_dir=paths.train_entities,
    processor=processor,
    max_seq_len=cfg["model"]["max_seq_len"],
    stems_subset=train_stems,
    )

    val_ds = ReceiptLayoutLMv3Dataset(
        img_dir=paths.train_img,
        box_dir=paths.train_box,
        ent_dir=paths.train_entities,
        processor=processor,
        max_seq_len=cfg["model"]["max_seq_len"],
        stems_subset=val_stems,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    out_ckpt = os.path.join(paths.work_dir, "checkpoints")

    # 🔹 Read from YAML whether to keep only the best model
    save_best = cfg["model"].get("save_best_only", False)

    args_train = TrainingArguments(
        output_dir=out_ckpt,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

        learning_rate=cfg["model"]["lr"],
        num_train_epochs=cfg["model"]["epochs"],
        weight_decay=cfg["model"]["weight_decay"],
        warmup_ratio=0.1,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        save_total_limit=2,

        # ✅ new: respect YAML setting
        load_best_model_at_end=save_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # ✅ ensures lower eval_loss = better

        # memory stability
        fp16=False,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        remove_unused_columns=True,
        report_to="none",
    )



    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,     # <-- use our dataset directly
        eval_dataset=val_ds,        # <-- use our dataset directly
        tokenizer=processor.tokenizer,
        data_collator=identity_collate,             # <-- important
        compute_metrics=compute_metrics_builder(id2label),
    )

        # --- Training ---
    train_output = trainer.train()
    print("Training done. Running final eval...")

    # --- Evaluate final model ---
    metrics = trainer.evaluate()
    print(metrics)

    # --- 🔹 Show the best checkpoint info if 'save_best_only' is enabled ---
    best_ckpt = None
    if hasattr(trainer, "state") and getattr(trainer.state, "best_model_checkpoint", None):
        best_ckpt = trainer.state.best_model_checkpoint
        best_metric = trainer.state.best_metric
        print(f"\n🏆 Best model selected from checkpoint: {best_ckpt}")
        print(f"   → Best eval_loss: {best_metric:.6f}\n")
    else:
        print("\nℹ️ No specific 'best' checkpoint was found (save_best_only disabled).")

    # --- Save final model and processor ---
    final_dir = os.path.join(paths.work_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print(f"Saved final model+processor to: {final_dir}")

    # --- 🧹 Optional cleanup ---
    if save_best and best_ckpt:
        ckpt_root = os.path.dirname(best_ckpt)
        for sub in os.listdir(ckpt_root):
            full = os.path.join(ckpt_root, sub)
            if os.path.isdir(full) and full != best_ckpt:
                import shutil
                shutil.rmtree(full, ignore_errors=True)
        print(f"🧹 Cleaned up all non-best checkpoints under: {ckpt_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--split_dir", default="", help="Use an existing split directory (with train.txt/val.txt)")
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()
    main(args)
