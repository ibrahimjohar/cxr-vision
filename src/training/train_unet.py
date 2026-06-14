#the unet localizes lung opacity regions. masks come from the rsna bounding boxes
#(scaled from 1024 to IMAGE_SIZE), painted as white rectangles on a black background.
#this is opacity-region localization, not pixel-perfect lesion segmentation - the boxes are
#coarse rectangles, so even a perfect prediction matches a rough region. dice in the 0.3-0.5
#range is expected and consistent with rsna segmentation baselines.

import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import (UNET_EPOCHS, UNET_LR, UNET_FEATURES, CHECKPOINTS_DIR, FIGURES_DIR, SEED, DEVICE)
from src.models.attention_unet import AttentionUNet
from src.utils.seg_dataset import get_seg_dataloaders


def dice_loss(pred, target, smooth=1.0):
    #soft dice over the batch. measures overlap between predicted and target mask.
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target):
    #bce drives per-pixel correctness, dice drives region overlap and handles the heavy
    #black/white imbalance (most pixels are background).
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def dice_score(pred, target, smooth=1.0, thresh=0.5):
    #hard dice at a 0.5 threshold - the actual segmentation quality metric we report.
    pred = (pred > thresh).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return ((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  validating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            total_loss += combined_loss(preds, masks).item() * images.size(0)
            #dice averaged per-image so empty-mask normals don't dominate the score
            for i in range(images.size(0)):
                total_dice += dice_score(preds[i], masks[i])

    N = len(loader.dataset)
    return total_loss / N, total_dice / N


def save_predictions(model, loader, device, epoch, n=4):
    #show input, ground-truth boxes, and predicted mask side by side.
    #prefer a batch that actually contains opacities so the figure isn't all-black.
    model.eval()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    chosen = None
    with torch.no_grad():
        for batch in loader:
            if batch["mask"].view(batch["mask"].size(0), -1).sum(dim=1).gt(0).any():
                chosen = batch
                break
    if chosen is None:
        chosen = next(iter(loader))

    images = chosen["image"][:n].to(device)
    masks = chosen["mask"][:n]
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

    plt.suptitle(f"unet opacity localization - epoch {epoch}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"unet_masks_epoch{epoch:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved predictions -> {path}")


def train(data_dir=None):
    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print("[data] loading dataset...")
    train_loader, val_loader, _ = get_seg_dataloaders(data_dir)

    model = AttentionUNet(in_channels=1, num_classes=1, features=UNET_FEATURES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=UNET_LR)
    print(f"[model] attention unet | params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_dice = 0.0
    print(f"\n[training] {UNET_EPOCHS} epochs\n")

    for epoch in range(1, UNET_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, device)

        print(f"epoch [{epoch:3d}/{UNET_EPOCHS}] train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | dice: {val_dice:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            save_predictions(model, val_loader, device, epoch)

        #checkpoint on dice (the segmentation quality metric), not loss
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": val_dice,
            }, os.path.join(CHECKPOINTS_DIR, "best_unet.pth"))
            print(f"saved best unet -> checkpoint (dice={val_dice:.4f})")

    print(f"\ndone. best val dice: {best_val_dice:.4f}")
    return model


if __name__ == "__main__":
    train()