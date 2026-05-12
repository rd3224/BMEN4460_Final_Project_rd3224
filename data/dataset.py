"""
ChestX-ray14 Dataset
NIH official patient-wise split  (train_val_list_subset.txt / test_list.txt)
Labels are read from Data_Entry_2017.csv; txt files only determine the split.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T

import config


# ── Label map ──────────────────────────────────────────────────────────────────
CLASS2IDX = {c: i for i, c in enumerate(config.CLASSES)}


def load_label_map() -> dict:
    """Return {filename: [label, ...]} from Data_Entry_2017.csv."""
    df = pd.read_csv(config.ENTRY_CSV, usecols=["Image Index", "Finding Labels"])
    label_map = {}
    for _, row in df.iterrows():
        labels = [l.strip() for l in str(row["Finding Labels"]).split("|")]
        label_map[row["Image Index"]] = labels
    return label_map


def parse_list_file(list_path: str, label_map: dict) -> pd.DataFrame:
    """Read txt (one filename per line) and join with label_map for labels."""
    rows = []
    with open(list_path) as f:
        for line in f:
            fname = line.strip()
            if not fname:
                continue
            labels = label_map.get(fname, ["No Finding"])
            rows.append({"filename": fname, "labels": labels})
    return pd.DataFrame(rows)


def labels_to_vec(labels):
    """Convert list of disease strings → 14-d binary vector."""
    vec = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    for lbl in labels:
        lbl = lbl.strip()
        if lbl in CLASS2IDX:
            vec[CLASS2IDX[lbl]] = 1.0
    return vec


# ── Transforms ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_train_transform():
    """Resize → random crop 224×224 → random H-flip → normalize."""
    return T.Compose([
        T.Resize(config.RESIZE_SIZE),
        T.RandomCrop(config.IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_test_transform():
    """Resize → center crop 224×224 → normalize  (paper §IV-C testing)."""
    return T.Compose([
        T.Resize(config.RESIZE_SIZE),
        T.CenterCrop(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_ten_crop_transform():
    """Five-crop × 2 flips for TTA (not used by default)."""
    return T.Compose([
        T.Resize(config.RESIZE_SIZE),
        T.TenCrop(config.IMAGE_SIZE),
        T.Lambda(lambda crops: torch.stack([
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(T.ToTensor()(c))
            for c in crops
        ])),
    ])


# ── Dataset ────────────────────────────────────────────────────────────────────
class ChestXray14(Dataset):
    """
    Parameters
    ----------
    list_path : str   Path to NIH split txt file (one filename per line).
    image_dir : str   Root directory containing all images.
    transform : callable, optional
    """

    def __init__(self, list_path: str, image_dir: str, transform=None):
        label_map = load_label_map()
        self.df = parse_list_file(list_path, label_map)
        self.image_dir = image_dir
        self.transform = transform or build_test_transform()

        self.targets = np.stack(
            [labels_to_vec(row.labels) for row in self.df.itertuples()],
            axis=0,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(self.targets[idx])
        return image, label, row["filename"]

    # ── Helpers ────────────────────────────────────────────────────────────────
    def get_label_counts(self):
        pos = self.targets.sum(axis=0)
        neg = len(self.targets) - pos
        return pos.astype(np.float32), neg.astype(np.float32)

    def class_weights(self):
        """βP = (|P|+|N|)/|P|,  βN = (|P|+|N|)/|N|   (paper eq. 4)"""
        pos, neg = self.get_label_counts()
        total = pos + neg
        beta_pos = total / (pos + 1e-6)
        beta_neg = total / (neg + 1e-6)
        return beta_pos, beta_neg


# ── Convenience factory ────────────────────────────────────────────────────────
def get_loaders(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    n_full = sum(1 for l in open(config.TRAIN_LIST) if l.strip())
    val_size   = int(n_full * config.VAL_SPLIT)
    train_size = n_full - val_size

    # Same seed → identical split indices for both datasets
    generator = torch.Generator().manual_seed(config.SEED)
    train_ds = ChestXray14(config.TRAIN_LIST, config.IMAGE_DIR, build_train_transform())
    train_subset, _ = random_split(train_ds, [train_size, val_size], generator=generator)

    generator_val = torch.Generator().manual_seed(config.SEED)
    val_base = ChestXray14(config.TRAIN_LIST, config.IMAGE_DIR, build_test_transform())
    _, val_subset = random_split(val_base, [train_size, val_size], generator=generator_val)

    test_ds = ChestXray14(config.TEST_LIST, config.IMAGE_DIR, build_test_transform())

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds
