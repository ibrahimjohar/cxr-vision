#GAN training is fundamentally different from supervised training - there's no ground truth loss to minimize.
#instead, two networks compete: D tries to distinguish real from fake, G tries to fool D.
#healthy training keeps D(x) ~ 0.5-0.7 and D(G(z)) ~ 0.3-0.5 throughout.
#if D(x) → 1 and D(G(z)) → 0, discriminator has won, generator gradients vanish (mode collapse).

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

from configs.config import (GAN_EPOCHS, GAN_LR_G, GAN_LR_D, GAN_BETA1, GAN_LATENT_DIM, CHECKPOINTS_DIR, FIGURES_DIR, IMAGE_SIZE, BATCH_SIZE, SEED, DEVICE)
from src.models.gan import build_gan, generator_loss, discriminator_loss
from src.utils.dataset import build_rsna_dataset, ChestXrayDataset


def get_gan_transforms():
    #gan output is tanh → [-1, 1], so real images need the same normalization
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] → [-1,1]
    ])


def get_gan_dataloaders(data_dir=None, batch_size=BATCH_SIZE):
    train_p, _, _, train_l, _, _ = build_rsna_dataset(data_dir)
    transform = get_gan_transforms()
    train_ds = ChestXrayDataset(train_p, train_l, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader


def train_one_epoch(G, D, loader, opt_g, opt_d, device, latent_dim):
    G.train()
    D.train()
    
    total_g_loss, total_d_loss = 0.0, 0.0
    d_real_scores, d_fake_scores = [], []

    for batch in tqdm(loader, desc="  training", leave=False):
        real = batch["image"].to(device)
        b = real.size(0)
        z = torch.randn(b, latent_dim).to(device)

        #train discriminator - maximize log(D(x)) + log(1 - D(G(z)))
        fake = G(z).detach()        #detach so gradients don't flow into G here
        d_real = D(real)
        d_fake = D(fake)
        d_loss = discriminator_loss(d_real, d_fake)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        #train generator - maximize log(D(G(z))) using non-saturating loss
        z = torch.randn(b, latent_dim).to(device)
        fake = G(z)
        d_fake_for_g = D(fake)
        g_loss = generator_loss(d_fake_for_g)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        total_d_loss += d_loss.item() * b
        total_g_loss += g_loss.item() * b
        d_real_scores.append(d_real.mean().item())
        d_fake_scores.append(d_fake.mean().item())

    N = len(loader.dataset)
    avg_d_real = sum(d_real_scores) / len(d_real_scores)
    avg_d_fake = sum(d_fake_scores) / len(d_fake_scores)
    return total_g_loss / N, total_d_loss / N, avg_d_real, avg_d_fake


def save_samples(G, device, latent_dim, epoch, n=16):
    #fixed noise so we can visually track how generations evolve across epochs
    G.eval()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        fake = G(z).cpu()

    cols = 4
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flatten()):
        img = fake[i].squeeze()
        img = (img + 1) / 2  # [-1,1] → [0,1] for display
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.suptitle(f"gan samples - epoch {epoch}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"gan_samples_epoch{epoch:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved samples -> {path}")


def train(data_dir=None):
    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print("[data] loading dataset...")
    train_loader = get_gan_dataloaders(data_dir)

    G, D = build_gan(latent_dim=GAN_LATENT_DIM, device=device)

    #dcgan paper recommends Adam with beta1=0.5 for stable adversarial training
    opt_g = optim.Adam(G.parameters(), lr=GAN_LR_G, betas=(GAN_BETA1, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=GAN_LR_D, betas=(GAN_BETA1, 0.999))

    
    print(f"\n[training] {GAN_EPOCHS} epochs\n")

    for epoch in range(1, GAN_EPOCHS + 1):
        g_loss, d_loss, d_real, d_fake = train_one_epoch(G, D, train_loader, opt_g, opt_d, device, GAN_LATENT_DIM)

        #d_real should stay ~0.5-0.7, d_fake ~0.3-0.5 - large divergence = mode collapse
        print(f"epoch [{epoch:3d}/{GAN_EPOCHS}] g_loss: {g_loss:.4f} | d_loss: {d_loss:.4f} | D(x): {d_real:.3f} D(G(z)): {d_fake:.3f}")

        if epoch % 5 == 0 or epoch == 1:
            save_samples(G, device, GAN_LATENT_DIM, epoch)

        # gans have no reliable quality metric to early-stop or pick "best" on, so we
        # checkpoint periodically and choose the epoch by inspecting the sample grids.
        if epoch % 5 == 0 or epoch == GAN_EPOCHS:
            torch.save({
                "epoch": epoch,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "opt_g_state": opt_g.state_dict(),
                "opt_d_state": opt_d.state_dict(),
            }, os.path.join(CHECKPOINTS_DIR, f"gan_epoch{epoch:03d}.pth"))
            print(f"  saved checkpoint at epoch {epoch}")

    print(f"\ndone. checkpoints saved every 5 epochs — pick best by inspecting sample grids")
    return G, D


if __name__ == "__main__":
    train()