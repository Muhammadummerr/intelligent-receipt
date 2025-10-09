import os, yaml, argparse

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from transformers import (
    LayoutLMv3Processor,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from seqeval.metrics import f1_score, classification_report
import torch

from ..data.dataset_layoutlmv3 import ReceiptLayoutLMv3Dataset, label_mappings
from ..data.splits import list_stems, make_train_val, save_split, load_split

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
        preds, labels = eval_pred
        # preds: (bsz, seq, nlabels), labels: (bsz, seq)
        pred_ids = np.argmax(preds, axis=-1)

        # convert to label strings, ignoring -100
        true_labels = []
        true_preds = []
        for p, l in zip(pred_ids, labels):
            tl = []
            tp = []
            for pi, li in zip(p, l):
                if li == -100:
                    continue
                tl.append(id2label[int(li)])
                tp.append(id2label[int(pi)])
            true_labels.append(tl)
            true_preds.append(tp)

        f1 = f1_score(true_labels, true_preds)
        return {"seqeval_f1": f1}
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

    hf_train = as_hf_dataset(train_ds)
    hf_val = as_hf_dataset(val_ds)

    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # training args (Kaggle-friendly defaults)
    out_ckpt = os.path.join(paths.work_dir, "checkpoints")
    args_train = TrainingArguments(
        output_dir=out_ckpt,
        per_device_train_batch_size=cfg["model"]["batch_size"],
        per_device_eval_batch_size=cfg["model"]["batch_size"],
        learning_rate=cfg["model"]["lr"],
        num_train_epochs=cfg["model"]["epochs"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,  # important for LayoutLMv3 inputs
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()
    print("Training done. Running final eval...")
    metrics = trainer.evaluate()
    print(metrics)

    # Save final
    final_dir = os.path.join(paths.work_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    processor.tokenizer.save_pretrained(final_dir)
    print(f"Saved final model to: {final_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--split_dir", default="", help="Use an existing split directory (with train.txt/val.txt)")
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()
    main(args)
