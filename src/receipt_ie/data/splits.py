import os, random, json
from typing import List, Tuple

def list_stems(img_dir: str, exts=(".jpg",".jpeg",".png")) -> List[str]:
    stems = []
    for fn in os.listdir(img_dir):
        base, ext = os.path.splitext(fn)
        if ext.lower() in exts:
            stems.append(base)
    return sorted(stems)

def make_train_val(stems: List[str], val_frac: float = 0.15, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    stems = stems[:]
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * val_frac))
    val = sorted(stems[:n_val])
    train = sorted(stems[n_val:])
    return train, val

def save_split(train_stems: List[str], val_stems: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_stems))
    with open(os.path.join(out_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_stems))
    # also JSON for convenience
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"train": train_stems, "val": val_stems}, f, indent=2)

def load_split(split_dir: str) -> Tuple[List[str], List[str]]:
    with open(os.path.join(split_dir, "train.txt")) as f:
        train = [l.strip() for l in f if l.strip()]
    with open(os.path.join(split_dir, "val.txt")) as f:
        val = [l.strip() for l in f if l.strip()]
    return train, val
