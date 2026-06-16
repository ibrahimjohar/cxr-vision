#trains the ddpm denoiser on all chest x-rays. like the gan, there's no clean validation
#metric for generation quality, so we checkpoint periodically and judge by the sample grids.
#images are normalized to [-1,1] to match the gaussian noise the diffusion process assumes.
#samples are generated with ddim (50 steps) every few epochs to watch fidelity improve.

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

from configs.config import (DIFFUSION_IMG_SIZE, DIFFUSION_TIMESTEPS, DIFFUSION_BASE_CH, DIFFUSION_EPOCHS, DIFFUSION_LR, DIFFUSION_BATCH, DIFFUSION_DDIM_STEPS, CHECKPOINTS_DIR, FIGURES_DIR, SEED, DEVICE)
from src.models.diffusion import UNet, Diffusion
from src.utils.dataset import build_rsna_dataset, ChestXrayDataset


def get_transforms():
    #grayscale [-1,1] to match the noise distribution the diffusion process assumes
    return transforms.Compose([
        transforms.Resize((DIFFUSION_IMG_SIZE, DIFFUSION_IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def get_dataloader(data_dir=None):
    train_p, _, _, train_l, _, _ = build_rsna_dataset(data_dir)
    ds = ChestXrayDataset(train_p, train_l, transform=get_transforms())
    return DataLoader(ds, batch_size=DIFFUSION_BATCH, shuffle=True, num_workers=0, pin_memory=True)


def save_samples(model, diffusion, epoch, n=16):
    samples = diffusion.ddim_sample(model, n, ddim_steps=DIFFUSION_DDIM_STEPS).cpu()
    samples = (samples + 1) / 2  # [-1,1] -> [0,1]

    os.makedirs(FIGURES_DIR, exist_ok=True)
    cols = 4
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.axis("off")
    plt.suptitle(f"diffusion samples - epoch {epoch}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"diffusion_samples_epoch{epoch:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved samples -> {path}")


def train(data_dir=None):
    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print("[data] loading dataset...")
    loader = get_dataloader(data_dir)

    model = UNet(base=DIFFUSION_BASE_CH).to(device)
    diffusion = Diffusion(DIFFUSION_TIMESTEPS, DIFFUSION_IMG_SIZE, device)
    optimizer = optim.Adam(model.parameters(), lr=DIFFUSION_LR)
    print(f"[model] ddpm unet | params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[config] img={DIFFUSION_IMG_SIZE} timesteps={DIFFUSION_TIMESTEPS} batch={DIFFUSION_BATCH}")

    print(f"\n[training] {DIFFUSION_EPOCHS} epochs\n")
    for epoch in range(1, DIFFUSION_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="  training", leave=False):
            images = batch["image"].to(device)
            optimizer.zero_grad()
            loss = diffusion.p_losses(model, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg = total_loss / len(loader.dataset)
        print(f"epoch [{epoch:3d}/{DIFFUSION_EPOCHS}] noise_mse: {avg:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            save_samples(model, diffusion, epoch)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, os.path.join(CHECKPOINTS_DIR, f"diffusion_epoch{epoch:03d}.pth"))
            print(f"  saved checkpoint at epoch {epoch}")

    print(f"\ndone. checkpoints saved every 5 epochs - pick best by inspecting sample grids")
    return model


if __name__ == "__main__":
    train()