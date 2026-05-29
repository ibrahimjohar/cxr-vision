"""
Standard PyTorch training loop for the CNN classifier.
Implements:
    - forward pass
    - loss computation
    - optimizer.zero_grad()
    - loss.backward()
    - optimizer.step()
    - learning rate scheduling
    - validation loop
    - checkpoint saving
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from configs.config import (LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,CHECKPOINTS_DIR, LOGS_DIR, DEVICE, SEED)
from src.models.classifier import build_classifier
from src.utils.dataset import get_dataloaders


#reproducibility
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#device setup
def get_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[Device] CUDA not available, using CPU.")
    return torch.device("cpu")


#training epoch
#returns avg loss & accuracy for this epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    #enables dropout, BatchNorm in training mode
    model.train()                       
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        #core training loop
        optimizer.zero_grad()               # 1. clear accumulated gradients
        outputs = model(images)             # 2. forward pass
        loss = criterion(outputs, labels)   # 3. compute loss
        loss.backward()                     # 4. backprop — compute gradients
        optimizer.step()                    # 5. update weights

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


#validation epoch
#evaludate model on validation set
#no gradient computation — model.eval() disables dropout and uses running BatchNorm stats (not batch stats)
def validate(model, loader, criterion, device):
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


#training loop with checkpointing and early stopping awareness
def train(backbone="resnet18", data_dir=None):
    set_seed()
    device = get_device()
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    #data
    print("[Data] loading dataset...")
    train_loader, val_loader, _ = get_dataloaders(data_dir)

    #model
    model = build_classifier(backbone).to(device)

    #loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    #training
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n[Training] Starting for {NUM_EPOCHS} epochs...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
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

        #save best model (early stopping concept - keep best weights)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_{backbone}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  [Saved] best model → {ckpt_path}  (val_acc={val_acc:.4f})")

    print(f"\n[Done] best validation accuracy: {best_val_acc:.4f}")
    return model, history


if __name__ == "__main__":
    model, history = train(backbone="resnet18")
