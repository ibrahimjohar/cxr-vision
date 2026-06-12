"""
standard pytorch training loop for the cnn classifier.
uses progressive unfreezing (frozen backbone first, then fine-tune) and
early stopping to prevent the fast overfitting seen with full fine-tuning.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from configs.config import (LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, CHECKPOINTS_DIR, LOGS_DIR, DEVICE, SEED, FREEZE_BACKBONE)
from src.models.classifier import build_classifier
from src.utils.dataset import get_dataloaders


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[Device] CUDA not available, using CPU.")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()               # 1. clear accumulated gradients
        outputs = model(images)             # 2. forward pass
        loss = criterion(outputs, labels)   # 3. compute loss
        loss.backward()                     # 4. backprop
        optimizer.step()                    # 5. update weights

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    # eval mode disables dropout and uses running batchnorm stats
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def set_backbone_trainable(model, trainable):
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def train(backbone="resnet18", data_dir=None):
    set_seed()
    device = get_device()
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("[Data] loading dataset...")
    train_loader, val_loader, _ = get_dataloaders(data_dir)

    model = build_classifier(backbone).to(device)

    # start with frozen backbone — only the classification head trains for the first few epochs.
    # this lets the head adapt to x-ray features before we risk overfitting the whole network.
    unfreeze_epoch = 5
    if FREEZE_BACKBONE:
        set_backbone_trainable(model, False)
        print(f"[Freeze] backbone frozen — will unfreeze at epoch {unfreeze_epoch}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    patience = 5
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n[Training] Starting for {NUM_EPOCHS} epochs...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # unfreeze backbone partway through and rebuild optimizer to include its params
        if FREEZE_BACKBONE and epoch == unfreeze_epoch:
            set_backbone_trainable(model, True)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY)
            print(f"  [Unfrozen] backbone now fine-tuning at reduced lr")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_{backbone}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  [Saved] best model → {ckpt_path}  (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stop] no improvement for {patience} epochs. stopping.")
                break

    print(f"\n[Done] best validation accuracy: {best_val_acc:.4f}")
    return model, history


if __name__ == "__main__":
    model, history = train(backbone="resnet18")