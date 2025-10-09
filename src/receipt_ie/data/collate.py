# src/receipt_ie/data/collate.py

from typing import Any, Dict, List

def identity_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Minimal collator for Hugging Face Trainer when you're NOT using `datasets`.
    Assumes each item is a dict of tensors plus a few non-tensor fields (e.g., 'id').

    - Stacks tensor-like values with `torch.stack`.
    - Keeps non-tensor values as a list (so 'id' becomes a list of strings).
    """
    import torch

    if not batch:
        return {}

    out: Dict[str, Any] = {}
    keys = batch[0].keys()

    for k in keys:
        v0 = batch[0][k]
        # Heuristic: tensors have `.shape` and are torch.Tensor
        if hasattr(v0, "shape") and isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            # keep as a simple python list (e.g., ids)
            out[k] = [b[k] for b in batch]

    return out
