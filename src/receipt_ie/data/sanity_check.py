import argparse, os
from transformers import LayoutLMv3Processor
from .dataset_layoutlmv3 import ReceiptLayoutLMv3Dataset, label_mappings

def main(args):
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    ds = ReceiptLayoutLMv3Dataset(
        img_dir=os.path.join(args.root, "train", "img"),
        box_dir=os.path.join(args.root, "train", "box"),
        ent_dir=os.path.join(args.root, "train", "entities"),
        processor=processor,
        max_seq_len=args.max_len,
    )
    print(f"Dataset size: {len(ds)}")
    item = ds[0]
    L2I, I2L = label_mappings()
    # show some tokens + labels
    import torch
    input_ids = item["input_ids"]
    labels = item["labels"]
    words = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    pairs = [(w, I2L.get(int(l), "NA")) for w, l in zip(words[:60], labels[:60].tolist())]
    print("Sample tokens+labels (first 60):")
    for w, l in pairs:
        print(f"{w:15s} -> {l}")
    print("OK.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root with train/test subfolders")
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()
    main(args)
