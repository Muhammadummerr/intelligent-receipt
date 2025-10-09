import os
from dataclasses import dataclass

@dataclass
class DatasetLayout:
    root: str
    train_img: str
    train_box: str
    train_entities: str
    test_img: str
    test_box: str
    test_entities: str

def build_layout(root: str) -> DatasetLayout:
    return DatasetLayout(
        root=root,
        train_img=os.path.join(root, "train", "img"),
        train_box=os.path.join(root, "train", "box"),
        train_entities=os.path.join(root, "train", "entities"),
        test_img=os.path.join(root, "test", "img"),
        test_box=os.path.join(root, "test", "box"),
        test_entities=os.path.join(root, "test", "entities"),
    )

def assert_dataset_ok(layout: DatasetLayout):
    for p in [layout.train_img, layout.train_box, layout.train_entities,
              layout.test_img, layout.test_box, layout.test_entities]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Expected folder missing: {p}")
