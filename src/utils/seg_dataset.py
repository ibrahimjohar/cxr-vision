import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from configs.config import DATA_RAW, IMAGE_SIZE, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, SEED

#segmentation dataset for the attention u-net.
#unlike the classifier dataset (one label per image), this builds a binary mask per image from the rsna bounding boxes.
#the boxes mark lung opacity regions in the original 1024x1024 dicom space, so every coordinate is scaled to IMAGE_SIZE before being painted onto the mask.
#normal images (no boxes) get an all-black mask.

ORIG_SIZE = 1024            #rsna dicom resolution that the box coordinates are defined in


class ChestXraySegDataset(Dataset):
    #boxes_by_path: dict mapping image path -> list of (x, y, w, h) in original 1024 space
    def __init__(self, image_paths, boxes_by_path, augment=False):
        self.image_paths = image_paths
        self.boxes_by_path = boxes_by_path
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def _build_mask(self, path):
        #start all-black, paint white rectangles for each opacity box, scaling 1024 -> IMAGE_SIZE
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        scale = IMAGE_SIZE / ORIG_SIZE
        for (x, y, w, h) in self.boxes_by_path.get(path, []):
            x0 = int(round(x * scale))
            y0 = int(round(y * scale))
            x1 = int(round((x + w) * scale))
            y1 = int(round((y + h) * scale))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(IMAGE_SIZE, x1), min(IMAGE_SIZE, y1)
            mask[y0:y1, x0:x1] = 1.0
        return mask

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = self._build_mask(path)

        img_t = torch.from_numpy(image).unsqueeze(0)        # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)        # (1, H, W)

        #light augmentation: horizontal flip applied to image AND mask together so they stay aligned
        if self.augment and torch.rand(1).item() < 0.5:
            img_t = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t)

        return {"image": img_t, "mask": mask_t, "path": path}


def build_seg_dataset(data_dir=None):
    #returns train/val/test splits as (paths, boxes_by_path) for the segmentation task.
    #keeps every box row (no drop_duplicates) so multi-box patients keep all their regions.
    if data_dir is None:
        data_dir = os.path.join(DATA_RAW, "rsna")

    csv_path = os.path.join(data_dir, "stage_2_train_labels.csv")
    image_dir = os.path.join(data_dir, "stage_2_train_images_png")
    df = pd.read_csv(csv_path)

    #group boxes per patient. normal rows have NaN coords and contribute no boxes.
    boxes_by_path = {}
    labels_by_path = {}
    for pid, group in df.groupby("patientId"):
        path = os.path.join(image_dir, f"{pid}.png")
        if not os.path.exists(path):
            continue
        boxes = []
        for _, row in group.iterrows():
            if row["Target"] == 1 and not pd.isna(row["x"]):
                boxes.append((float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])))
        boxes_by_path[path] = boxes
        labels_by_path[path] = 1 if boxes else 0

    paths = list(boxes_by_path.keys())
    strat = [labels_by_path[p] for p in paths]  # stratify on has-opacity so splits stay balanced

    train_p, temp_p, train_s, temp_s = train_test_split(paths, strat, test_size=(1 - TRAIN_SPLIT), random_state=SEED, stratify=strat)
    val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=SEED, stratify=temp_s)

    n_pos = sum(strat)
    print(f"[seg] total: {len(paths)} | with opacity: {n_pos} | normal: {len(paths)-n_pos}")
    print(f"[seg] train: {len(train_p)} | val: {len(val_p)} | test: {len(test_p)}")
    return train_p, val_p, test_p, boxes_by_path


def get_seg_dataloaders(data_dir=None, batch_size=BATCH_SIZE):
    train_p, val_p, test_p, boxes_by_path = build_seg_dataset(data_dir)

    train_ds = ChestXraySegDataset(train_p, boxes_by_path, augment=True)
    val_ds = ChestXraySegDataset(val_p, boxes_by_path, augment=False)
    test_ds = ChestXraySegDataset(test_p, boxes_by_path, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tl, vl, _ = get_seg_dataloaders()
    batch = next(iter(tl))
    print(f"image: {batch['image'].shape} | mask: {batch['mask'].shape}")
    print(f"mask coverage (mean white fraction): {batch['mask'].mean().item():.4f}")