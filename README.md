# Intelligent Receipt — Part 1 (Document Information Extraction)

Goal: extract `company`, `date`, `address`, `total` from scanned receipts.

## Structure
- `src/receipt_ie/` — package code (data, models, training, eval)
- `configs/default.yaml` — paths & hyperparams
- `notebooks/` — Kaggle/Colab notebooks for experiments
- `data/` — (ignored) put local copies here if needed
- `outputs/` — (ignored) checkpoints & logs

## Kaggle quickstart
1. Turn on Internet in your Kaggle Notebook.
2. `!git clone <your-repo-url>`  
3. `cd intelligent-receipt && pip install -r requirements.txt`
4. Attach your Kaggle dataset with folders: `train/img|box|entities`, `test/img|box|entities`.
5. Update `configs/default.yaml` `data.root` to `/kaggle/input/<your-dataset-name>`.

## License

