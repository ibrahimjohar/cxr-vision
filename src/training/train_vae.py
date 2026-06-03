import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import (VAE_EPOCHS, VAE_LR, VAE_BETA, CHECKPOINTS_DIR, FIGURES_DIR, IMAGE_SIZE, BATCH_SIZE, SEED, DEVICE)
from src.models.vae import VAE, vae_loss
from src.utils.dataset import build_rsna_dataset, ChestXrayDataset


def get_vae_transforms():
    #vae takes grayscale [0,1] - not imagenet normalized rgb like the classifier
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])


def get_vae_dataloaders(data_dir=None, batch_size=BATCH_SIZE):
    train_p, val_p, _, train_l, val_l, _ = build_rsna_dataset(data_dir)
    transform = get_vae_transforms()
    
    train_ds = ChestXrayDataset(train_p, train_l, transform=transform)
    val_ds = ChestXrayDataset(val_p, val_l, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    
    for batch in tqdm(loader, desc="  training", leave=False):
        images = batch["image"].to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(images)
        loss, recon_l, kl_l = vae_loss(reconstruction, images, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        n = images.size(0)
        total_loss += loss.item() * n
        total_recon += recon_l.item() * n
        total_kl += kl_l.item() * n
        
    N = len(loader.dataset)
    
    return total_loss / N, total_recon / N, total_kl / N


def validate(model, loader, device, beta):
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="  validating", leave=False):
            images = batch["image"].to(device)
            reconstruction, mu, logvar = model(images)          # encoder → reparameterize → decoder
            loss, recon_l, kl_l = vae_loss(reconstruction, images, mu, logvar, beta)
            n = images.size(0)
            total_loss += loss.item() * n
            total_recon += recon_l.item() * n
            total_kl += kl_l.item() * n
            
    N = len(loader.dataset)
    
    return total_loss / N, total_recon / N, total_kl / N


def save_samples(model, loader, device, epoch, n=8):
    #save original vs reconstruction side by side to visually check training progress
    model.eval()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    batch = next(iter(loader))
    images = batch["image"][:n].to(device)
    
    with torch.no_grad():
        recons, _, _ = model(images)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    
    for i in range(n):
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("reconstructed", fontsize=10)
    plt.suptitle(f"vae reconstructions — epoch {epoch}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"vae_recon_epoch{epoch:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved reconstructions -> {path}")


def train(data_dir=None):
    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print("[data] loading dataset...")
    train_loader, val_loader = get_vae_dataloaders(data_dir)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=VAE_LR)
    print(f"[model] vae | params: {sum(p.numel() for p in model.parameters()):,}")

    #track recon and kl separately - if kl explodes the model is memorizing not generalizing
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}

    print(f"\n[training] {VAE_EPOCHS} epochs | beta={VAE_BETA}\n")

    for epoch in range(1, VAE_EPOCHS + 1):
        train_loss, train_recon, train_kl = train_one_epoch(model, train_loader, optimizer, device, VAE_BETA)
        val_loss, _, _ = validate(model, val_loader, device, VAE_BETA)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["recon_loss"].append(train_recon)
        history["kl_loss"].append(train_kl)

        print(f"epoch [{epoch:3d}/{VAE_EPOCHS}] train: {train_loss:.4f} (recon={train_recon:.4f} kl={train_kl:.4f}) | val: {val_loss:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            save_samples(model, val_loader, device, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CHECKPOINTS_DIR, "best_vae.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"saved best vae -> {ckpt_path} (val_loss={val_loss:.4f})")

    print(f"\ndone. best val loss: {best_val_loss:.4f}")
    return model, history


if __name__ == "__main__":
    train()