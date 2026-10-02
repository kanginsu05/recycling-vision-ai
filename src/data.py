"""Data loading, caching, splitting and transforms.

All JPEGs are decoded once and cached as 256x256 uint8 arrays (data/cache/*.npy),
so every experiment reads exactly the same pixels and CPU time goes into the model,
not into JPEG decoding.
"""
import json

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .config import (CACHE_DIR, CACHE_SIZE, CLASSES, HOLDOUT_FRAC, IMAGENET_MEAN,
                     IMAGENET_STD, RESULTS_DIR, SEED, TEST_DIR, TRAIN_DIR)


def _load_image(path, size=CACHE_SIZE):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_train():
    """Return (images[N,H,W,3] uint8, labels[N], paths[N]) for data/train."""
    x_path, y_path, p_path = (CACHE_DIR / f"train_{n}.npy" for n in ("x", "y", "paths"))
    if x_path.exists():
        return np.load(x_path), np.load(y_path), np.load(p_path)

    paths, labels = [], []
    for ci, cls in enumerate(CLASSES):
        for p in sorted((TRAIN_DIR / cls).glob("*.jpg")):
            paths.append(f"{cls}/{p.name}")
            labels.append(ci)
    x = np.stack([_load_image(TRAIN_DIR / p) for p in paths])
    y, paths = np.array(labels, dtype=np.int64), np.array(paths)
    np.save(x_path, x); np.save(y_path, y); np.save(p_path, paths)
    return x, y, paths


def load_test():
    """Return (images[N,H,W,3] uint8, file_names[N]) for the unlabeled Kaggle test set."""
    x_path, n_path = CACHE_DIR / "test_x.npy", CACHE_DIR / "test_names.npy"
    if x_path.exists():
        return np.load(x_path), np.load(n_path)
    names = sorted(p.name for p in TEST_DIR.glob("*.jpg"))
    x = np.stack([_load_image(TEST_DIR / n) for n in names])
    names = np.array(names)
    np.save(x_path, x); np.save(n_path, names)
    return x, names


def holdout_split(labels):
    """Fixed stratified dev/hold-out split shared by every experiment.

    The hold-out part is never used for training, model selection or tuning;
    it is only used to report final numbers.
    """
    split_file = RESULTS_DIR / "split.json"
    if split_file.exists():
        s = json.loads(split_file.read_text())
        return np.array(s["dev"]), np.array(s["holdout"])
    dev, hold = train_test_split(np.arange(len(labels)), test_size=HOLDOUT_FRAC,
                                 stratify=labels, random_state=SEED)
    dev, hold = np.sort(dev), np.sort(hold)
    split_file.write_text(json.dumps({"seed": SEED, "holdout_frac": HOLDOUT_FRAC,
                                      "dev": dev.tolist(), "holdout": hold.tolist()}))
    return dev, hold


class ArrayDataset(Dataset):
    """Wraps cached uint8 HWC images; applies a torchvision v2 transform per sample."""

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1)  # CHW uint8
        if self.transform is not None:
            x = self.transform(x)
        y = -1 if self.labels is None else int(self.labels[i])
        return x, y


def resize_images(images, size):
    """Resize a whole uint8 NHWC array once (cheaper than resizing every epoch)."""
    return np.stack([np.asarray(Image.fromarray(im).resize((size, size), Image.BILINEAR))
                     for im in images])


# ---------------------------------------------------------------- transforms
def baseline_transform(size, augment=False, pre_resized=False):
    """Transforms used in the original competition notebook (no normalization)."""
    ops = [] if pre_resized else [T.Resize((size, size), antialias=True)]
    if augment:
        ops += [T.RandomHorizontalFlip(0.5), T.RandomRotation(15)]
    ops += [T.ToDtype(torch.float32, scale=True)]
    return T.Compose(ops)


def pretrained_transform(size=224, augment=False):
    """ImageNet-style transforms for pretrained backbones."""
    if augment:
        ops = [
            T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(20, fill=255),  # backgrounds are white-ish
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        ]
    else:
        ops = [T.Resize((size, size), antialias=True)]
    ops += [T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)
