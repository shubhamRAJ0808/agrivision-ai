"""
Lightweight image dataset loader (PIL + numpy + torch only, no torchvision).
"""
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_SIZE = 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_as_tensor(path_or_pil, train=False, size=IMG_SIZE):
    """Load an image (path or PIL.Image), resize, augment (if train), normalize, return CHW tensor."""
    img = Image.open(path_or_pil).convert("RGB") if isinstance(path_or_pil, str) else path_or_pil.convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)

    if train:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            if angle:
                img = img.rotate(angle)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(arr.copy())


class LeafDataset(Dataset):
    """Expects a root dir with one subfolder per class, containing images."""

    def __init__(self, root, classes, samples, train=False):
        self.root = root
        self.classes = classes
        self.samples = samples  # list of (relative_path, label_idx)
        self.train = train

    @classmethod
    def from_folder(cls, root, val_split=0.15, seed=42):
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        rng = random.Random(seed)
        train_samples, val_samples = [], []
        for label_idx, class_name in enumerate(classes):
            files = sorted(os.listdir(os.path.join(root, class_name)))
            rng.shuffle(files)
            n_val = max(1, int(len(files) * val_split))
            val_files, train_files = files[:n_val], files[n_val:]
            train_samples += [(os.path.join(class_name, f), label_idx) for f in train_files]
            val_samples += [(os.path.join(class_name, f), label_idx) for f in val_files]
        return (
            cls(root, classes, train_samples, train=True),
            cls(root, classes, val_samples, train=False),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        tensor = load_image_as_tensor(os.path.join(self.root, rel_path), train=self.train)
        return tensor, label
