# src/receipt_ie/data/collate.py

from typing import Any, Dict, List

def identity_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collator for HF Trainer when not using 🤗 datasets.
    Returns ONLY the keys the model expects; drops metadata like 'id'.

    Expected tensor keys for LayoutLMv3 token classification:
      - input_ids
      - attention_mask
      - bbox
      - pixel_values
      - labels
      - (optional) token_type_ids
      d
    """
    import torch

    if not batch:
        return {}

    allowed = {
        "input_ids",
        "attention_mask",
        "bbox",
        "pixel_values",
        "labels",
        "token_type_ids",  # may or may not be present
    }

    out: Dict[str, Any] = {}
    # stack only allowed tensor keys that exist
    for k in allowed:
        if k in batch[0] and hasattr(batch[0][k], "shape") and isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    #NOTE: we intentionally DROP non-tensor metadata (e.g., 'id').
    return out
