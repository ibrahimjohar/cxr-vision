"""
src/utils/dataset.py

Handles dataset loading, splitting, and PyTorch DataLoader creation.
Works with RSNA Pneumonia dataset out of the box.
Swap dataset_name in config to extend to ChestX-ray14.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from configs.config import (
    DATA_RAW, IMAGE_SIZE, BATCH_SIZE,
    TRAIN_SPLIT, VAL_SPLIT, SEED, NUM_CLASSES
)


# ─── Custom Dataset Class ─────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    Generic chest X-ray dataset.
    Expects:
        image_paths : list of full paths to images
        labels      : list of integer labels (0 = normal, 1 = pneumonia)
        transform   : torchvision transform pipeline
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image as RGB (even if grayscale X-ray — ResNet expects 3 channels)
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"image": image, "label": label, "path": self.image_paths[idx]}


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(mode="train"):
    """
    Returns transform pipeline for train / val / test.
    Training includes augmentation. Val/Test only normalize.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet mean (used because we use pretrained ResNet)
    std  = [0.229, 0.224, 0.225]   # ImageNet std

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_rsna_dataset(data_dir=None):
    """
    Builds image path + label lists from the RSNA Pneumonia dataset.
    Expected folder structure after download:
        data/raw/rsna/
            stage_2_train_labels.csv
            stage_2_train_images/
                *.dcm  (or *.png if pre-converted)

    Returns:
        train_paths, val_paths, test_paths : lists of image paths
        train_labels, val_labels, test_labels : lists of int labels
    """
    if data_dir is None:
        data_dir = os.path.join(DATA_RAW, "rsna")

    csv_path = os.path.join(data_dir, "stage_2_train_labels.csv")

    if not os.path.exists(csv_path):
        print(f"[WARNING] RSNA labels CSV not found at {csv_path}")
        print("Using synthetic dummy data for scaffold testing.")
        return _make_dummy_data()

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="patientId")   # one row per patient

    image_dir = os.path.join(data_dir, "stage_2_train_images_png")
    df["path"] = df["patientId"].apply(
        lambda pid: os.path.join(image_dir, f"{pid}.png")
    )
    df = df[df["path"].apply(os.path.exists)]     # keep only existing files

    paths  = df["path"].tolist()
    labels = df["Target"].tolist()                # 0 = normal, 1 = pneumonia

    return _split_data(paths, labels)


def _split_data(paths, labels):
    """Train / val / test split with stratification."""
    # First split: train vs temp (val+test)
    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels,
        test_size=(1 - TRAIN_SPLIT),
        random_state=SEED,
        stratify=labels
    )
    # Second split: val vs test from temp
    val_ratio = VAL_SPLIT / (VAL_SPLIT + (1 - TRAIN_SPLIT - VAL_SPLIT))
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_l
    )
    print(f"[Dataset] Train: {len(train_p)} | Val: {len(val_p)} | Test: {len(test_p)}")
    return train_p, val_p, test_p, train_l, val_l, test_l


def _make_dummy_data(n=200):
    """Synthetic dummy data for testing the pipeline without real images."""
    import tempfile
    from PIL import Image as PILImage

    print("[INFO] Generating dummy PNG images for pipeline testing...")
    dummy_dir = os.path.join(DATA_RAW, "dummy")
    os.makedirs(dummy_dir, exist_ok=True)

    paths, labels = [], []
    for i in range(n):
        label = i % 2
        path  = os.path.join(dummy_dir, f"dummy_{i:04d}.png")
        if not os.path.exists(path):
            arr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            PILImage.fromarray(arr, mode='L').save(path)
        paths.append(path)
        labels.append(label)

    return _split_data(paths, labels)


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(data_dir=None, batch_size=BATCH_SIZE):
    """
    Main entry point. Returns train, val, test DataLoaders.
    Usage:
        from src.utils.dataset import get_dataloaders
        train_loader, val_loader, test_loader = get_dataloaders()
    """
    train_p, val_p, test_p, train_l, val_l, test_l = build_rsna_dataset(data_dir)

    train_ds = ChestXrayDataset(train_p, train_l, transform=get_transforms("train"))
    val_ds   = ChestXrayDataset(val_p,   val_l,   transform=get_transforms("val"))
    test_ds  = ChestXrayDataset(test_p,  test_l,  transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    batch = next(iter(train_loader))
    print(f"Batch image shape : {batch['image'].shape}")
    print(f"Batch label shape : {batch['label'].shape}")
    print(f"Labels in batch   : {batch['label']}")
