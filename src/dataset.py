"""
dataset.py
----------
PyTorch Dataset and DataLoader loaders with configurable augmentation pipelines.
Supports:
- Evaluation / Baseline transforms (Resize, CenterCrop, Normalize)
- Data Augmentation pipeline (RandomResizedCrop, Flips, Rotations, ColorJitter)
- Class weighting calculation for imbalanced CrossEntropyLoss
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str = "train", augment: bool = True, img_size: int = 224) -> transforms.Compose:
    """
    Returns data transforms for train/val/test splits.
    
    Data Augmentation Strategy:
    - RandomResizedCrop simulates varied distance and camera framing of trash objects.
    - RandomHorizontalFlip & RandomVerticalFlip account for arbitrary object orientation in waste bins.
    - RandomRotation (degrees=25) simulates tossed or tilted waste.
    - ColorJitter (brightness, contrast, saturation) accounts for indoor vs outdoor lighting fluctuations.
    """
    if split == "train" and augment:
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Deterministic transform for validation, testing, or non-augmented baseline
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class TrashNetDataset(Dataset):
    """PyTorch Dataset loading images based on splits manifest with in-memory caching."""
    def __init__(self, items: List[dict], data_root: Path, transform=None, preload: bool = True):
        self.items = items
        self.data_root = Path(data_root)
        self.transform = transform
        self.preload = preload
        self.cache = []

        if self.preload:
            # Pre-resize to 256x256 once to reduce memory and speed up crop transforms
            for item in items:
                img_path = self.data_root / item["rel_path"]
                with Image.open(img_path) as img:
                    self.cache.append(img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        if self.preload:
            img = self.cache[idx]
        else:
            img_path = self.data_root / item["rel_path"]
            img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        label = item["class_idx"]
        return img, label, item["rel_path"]


def compute_class_weights(train_items: List[dict], num_classes: int = 6) -> torch.Tensor:
    """
    Computes inverse-frequency class weights to counteract dataset imbalance:
    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = [0] * num_classes
    for item in train_items:
        counts[item["class_idx"]] += 1
    
    total = len(train_items)
    weights = [total / (num_classes * max(c, 1)) for c in counts]
    # Normalize weights so mean is 1.0
    mean_w = sum(weights) / num_classes
    norm_weights = [w / mean_w for w in weights]
    return torch.tensor(norm_weights, dtype=torch.float32)


def build_dataloaders(
    manifest_path: str = "data/splits.json",
    data_root: str = "data/dataset-resized",
    batch_size: int = 32,
    augment_train: bool = True,
    img_size: int = 224,
    num_workers: int = 0
) -> Tuple[Dict[str, DataLoader], List[str], torch.Tensor]:
    """
    Loads splits from JSON manifest and creates DataLoaders for train, val, and test.
    """
    manifest_p = Path(manifest_path)
    data_root_p = Path(data_root)

    with open(manifest_p, "r") as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    train_items = manifest["splits"]["train"]
    val_items = manifest["splits"]["val"]
    test_items = manifest["splits"]["test"]

    class_weights = compute_class_weights(train_items, num_classes=len(classes))

    train_ds = TrashNetDataset(
        items=train_items,
        data_root=data_root_p,
        transform=get_transforms("train", augment=augment_train, img_size=img_size)
    )
    val_ds = TrashNetDataset(
        items=val_items,
        data_root=data_root_p,
        transform=get_transforms("val", augment=False, img_size=img_size)
    )
    test_ds = TrashNetDataset(
        items=test_items,
        data_root=data_root_p,
        transform=get_transforms("test", augment=False, img_size=img_size)
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
    }

    return loaders, classes, class_weights
