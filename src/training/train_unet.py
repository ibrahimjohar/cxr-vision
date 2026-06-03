#the unet is trained for segmentation - localizing where pneumonia is in the image.
#ideally this requires pixel-level masks, but the rsna dataset only has bounding boxes.
#we convert bounding boxes to binary masks as a proxy. when no box exists (normal cases),
#the mask is all zeros. this is a weak supervision setup - common in medical imaging.

import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import (UNET_EPOCHS, UNET_LR, UNET_FEATURES, CHECKPOINTS_DIR, FIGURES_DIR, IMAGE_SIZE, BATCH_SIZE, SEED, DEVICE)
from src.models.attention_unet import AttentionUNet
from src.utils.dataset import build_rsna_dataset, ChestXrayDataset

def get_unet_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])


def get_unet_dataloaders(data_dir=None, batch_size=BATCH_SIZE):
    train_p, val_p, _, train_l, val_l, _ = build_rsna_dataset(data_dir)
    transform = get_unet_transforms()
    
    train_ds = ChestXrayDataset(train_p, train_l, transform=transform)
    val_ds = ChestXrayDataset(val_p, val_l, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader

#proxy masks from classification labels - pneumonia (1) = full white mask, normal (0) = black
#stand-in until real bounding box masks are implemented with the full rsna dataset
def make_masks_from_labels(labels, size=IMAGE_SIZE):
    b = labels.size(0)
    masks = torch.zeros(b, 1, size, size)
    for i in range(b):
        if labels[i].item() == 1:
            masks[i] = 1.0
    return masks

#dice loss measures overlap between predicted and target mask
#better than pure BCE for segmentation because it handles class imbalance
def dice_loss(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

#dice + bce together: dice handles overlap quality, bce handles pixel-level accuracy
def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"]
        masks = make_masks_from_labels(labels).to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  validating", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"]
            masks = make_masks_from_labels(labels).to(device)

            preds = model(images)
            loss = combined_loss(preds, masks)
            total_loss += loss.item() * images.size(0)
            total_dice += (1 - dice_loss(preds, masks).item()) * images.size(0)

    N = len(loader.dataset)
    return total_loss / N, total_dice / N

#visualize input, predicted mask, and ground truth side by side
def save_attention_maps(model, loader, device, epoch, n=4):
    model.eval()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    batch = next(iter(loader))
    images = batch["image"][:n].to(device)
    labels = batch["label"][:n]
    masks = make_masks_from_labels(labels)

    with torch.no_grad():
        preds = model(images).cpu()

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    for i in range(n):
        axes[i, 0].imshow(images[i].cpu().squeeze(), cmap="gray")
        axes[i, 0].set_title("input")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i].squeeze(), cmap="hot")
        axes[i, 1].set_title("ground truth")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(preds[i].squeeze(), cmap="hot")
        axes[i, 2].set_title("prediction")
        axes[i, 2].axis("off")

    plt.suptitle(f"unet predictions — epoch {epoch}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"unet_masks_epoch{epoch:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved attention maps -> {path}")


def train(data_dir=None):
    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print("[data] loading dataset...")
    train_loader, val_loader = get_unet_dataloaders(data_dir)

    model = AttentionUNet(in_channels=1, num_classes=1, features=UNET_FEATURES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=UNET_LR)
    print(f"[model] attention unet | params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")
    print(f"\n[training] {UNET_EPOCHS} epochs\n")

    for epoch in range(1, UNET_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, device)

        print(f"epoch [{epoch:3d}/{UNET_EPOCHS}] train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | dice: {val_dice:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            save_attention_maps(model, val_loader, device, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
            }, os.path.join(CHECKPOINTS_DIR, "best_unet.pth"))
            print(f"saved best unet -> checkpoint (val_loss={val_loss:.4f} dice={val_dice:.4f})")

    print(f"\ndone. best val loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    train()