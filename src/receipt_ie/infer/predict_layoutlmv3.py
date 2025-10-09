# src/receipt_ie/infer/predict_layoutlmv3.py
import os, json, argparse, torch, numpy as np
from tqdm import tqdm
from transformers import LayoutLMv3Processor, AutoModelForTokenClassification
from ..data.dataset_infer import ReceiptInferenceDataset
from ..data.collate import identity_collate
from ..utils.text import split_tokens
from ..utils.postproc import group_bio, norm_spaces, norm_date, norm_total

@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    processor = LayoutLMv3Processor.from_pretrained(args.model_dir, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval().to(args.device)

    ds = ReceiptInferenceDataset(
        img_dir=os.path.join(args.data_root, "test", "img"),
        box_dir=os.path.join(args.data_root, "test", "box"),
        processor=processor,
        max_seq_len=args.max_seq_len,
    )

    # simple DataLoader
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=identity_collate)

    # we also need the stems (ids) to name files; get in parallel via ds.stems
    stems = ds.stems
    ptr = 0

    all_out = []
    for batch in tqdm(dl, total=(len(ds)+args.batch_size-1)//args.batch_size):
        # forward
        batch_on = {k: v.to(args.device) for k, v in batch.items()}
        logits = model(**batch_on).logits  # (B, seq, n_labels)
        pred_ids = logits.argmax(dim=-1).cpu().numpy()

        # decode words back from input_ids (fast & robust)
        input_ids = batch["input_ids"].cpu().numpy()
        for i in range(pred_ids.shape[0]):
            sid = stems[ptr]; ptr += 1
            # convert ids to tokens, then to "words" (same tokenizer used)
            toks = processor.tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
            # drop padding pairs with attention_mask==0
            attn = batch["attention_mask"][i].cpu().numpy().tolist()
            toks = [t for t,a in zip(toks, attn) if a==1]

            # convert tokens (wordpiece) back to words we fed initially:
            # our inputs were `words` already, not raw text, so tokenizer is in "is_split_into_words" mode.
            # We can recover per-token word indices via word_ids() if we had the encoding object;
            # since we don't here, we approximate by merging '##' pieces.
            words = []
            buff = ""
            for t in toks:
                if t in ("[PAD]", "[CLS]", "[SEP]"):
                    continue
                if t.startswith("##"):
                    buff += t[2:]
                else:
                    if buff:
                        words.append(buff)
                    buff = t
            if buff: words.append(buff)

            # align pred_ids to same (drop pads too)
            pids = pred_ids[i][:len(toks)]

            # group BIO -> fields
            fields = group_bio(words, pids)

            # normalize some outputs
            fields_norm = {
                "company": norm_spaces(fields.get("company","")),
                "date":    norm_date(fields.get("date","")),
                "address": norm_spaces(fields.get("address","")),
                "total":   norm_total(fields.get("total","")),
            }

            rec = {"id": sid, **fields_norm}
            all_out.append(rec)

            # write per-file
            with open(os.path.join(args.out_dir, f"{sid}.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "company": fields_norm["company"],
                    "date":    fields_norm["date"],
                    "address": fields_norm["address"],
                    "total":   fields_norm["total"],
                }, f, ensure_ascii=False, indent=2)

    # also dump a combined file
    with open(os.path.join(args.out_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
        for r in all_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", default="./preds")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args)
