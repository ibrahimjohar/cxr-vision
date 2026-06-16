#computes fid to compare gan vs diffusion for synthetic chest x-ray quality.
#fid measures the distance between real and generated image distributions in
#inception-v3 feature space. lower fid = generated images closer to real distribution.
#important caveat: inception was trained on imagenet (natural color images), not x-rays.
#so absolute fid values are inflated/unreliable for medical images. what IS meaningful
#is the RELATIVE comparison - if diffusion fid < gan fid by a meaningful margin, diffusion
#produces more realistic-looking images in feature space. we report it this way.

import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import (CHECKPOINTS_DIR, FIGURES_DIR, DEVICE, GAN_LATENT_DIM, DIFFUSION_IMG_SIZE, DIFFUSION_TIMESTEPS, DIFFUSION_BASE_CH, DIFFUSION_DDIM_STEPS)
from src.models.gan import build_gan
from src.models.diffusion import UNet, Diffusion
from src.utils.dataset import build_rsna_dataset, ChestXrayDataset

N_SAMPLES = 500         #samples from each generative model
N_REAL = 500            #real images for reference distribution
FID_IMG_SIZE = 299      #inception expects 299x299


def get_inception():
    #using torchvision's inception_v3 in feature-extraction mode (no final classifier)
    from torchvision.models import inception_v3
    model = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


def to_inception_input(imgs_gray, device):
    #inception expects 3-channel 299x299 float images normalized to [-1,1].
    #our generated images are grayscale so we repeat channels.
    resize = transforms.Resize((FID_IMG_SIZE, FID_IMG_SIZE), antialias=True)
    imgs = imgs_gray.repeat(1, 3, 1, 1)   # (B,1,H,W) -> (B,3,H,W)
    imgs = resize(imgs)
    imgs = (imgs - 0.5) / 0.5             # [0,1] -> [-1,1]
    return imgs.to(device)


@torch.no_grad()
def extract_features(model, images, device, batch_size=32):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        batch = to_inception_input(batch, device)
        f = model(batch)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(real_feats, gen_feats):
    from scipy.linalg import sqrtm
    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sig_r = np.cov(real_feats, rowvar=False)
    sig_g = np.cov(gen_feats, rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sig_r + sig_g - 2 * covmean)
    return float(fid)


def get_real_images(n):
    _, _, test_p, _, _, test_l = build_rsna_dataset()
    transform = transforms.Compose([
        transforms.Resize((DIFFUSION_IMG_SIZE, DIFFUSION_IMG_SIZE)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    ds = ChestXrayDataset(test_p[:n], test_l[:n], transform=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    imgs = []
    for batch in loader:
        imgs.append(batch["image"])
    return torch.cat(imgs, dim=0)[:n]


@torch.no_grad()
def generate_gan(n, device):
    G, _ = build_gan(latent_dim=GAN_LATENT_DIM, device=device)
    ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "gan_epoch010.pth"), map_location=device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    print(f"[gan] loaded epoch {ckpt['epoch']} checkpoint")

    imgs = []
    batch_size = 32
    for i in tqdm(range(0, n, batch_size), desc="  generating gan samples"):
        b = min(batch_size, n - i)
        z = torch.randn(b, GAN_LATENT_DIM, device=device)
        fake = G(z)
        fake = (fake + 1) / 2   # [-1,1] -> [0,1]
        #gan outputs 224x224, resize to diffusion size for fair comparison
        fake = transforms.Resize((DIFFUSION_IMG_SIZE, DIFFUSION_IMG_SIZE), antialias=True)(fake)
        imgs.append(fake.cpu())
    return torch.cat(imgs, dim=0)[:n]


@torch.no_grad()
def generate_diffusion(n, device):
    model = UNet(base=DIFFUSION_BASE_CH).to(device)
    ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "diffusion_epoch010.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[diffusion] loaded epoch {ckpt['epoch']} checkpoint")

    diffusion = Diffusion(DIFFUSION_TIMESTEPS, DIFFUSION_IMG_SIZE, device)
    imgs = []
    batch_size = 16   #ddim sampling is memory-heavy
    for i in tqdm(range(0, n, batch_size), desc="  generating diffusion samples"):
        b = min(batch_size, n - i)
        samples = diffusion.ddim_sample(model, b, ddim_steps=DIFFUSION_DDIM_STEPS)
        samples = (samples + 1) / 2   # [-1,1] -> [0,1]
        imgs.append(samples.cpu())
    return torch.cat(imgs, dim=0)[:n]


def plot_results(gan_fid, diff_fid):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["dcgan", "ddpm (diffusion)"], [gan_fid, diff_fid], color=["steelblue", "tomato"], width=0.5)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylabel("fid score (lower = better)")
    ax.set_title("generative model comparison — fid vs real chest x-rays\n"
                 "(note: absolute values inflated by imagenet/medical domain mismatch)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "generative_fid_comparison.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved fid comparison -> {path}")


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    inception = get_inception().to(device)

    print(f"\n[real images] extracting features from {N_REAL} test images...")
    real_imgs = get_real_images(N_REAL)
    real_feats = extract_features(inception, real_imgs, device)

    print(f"\n[gan] generating {N_SAMPLES} samples...")
    gan_imgs = generate_gan(N_SAMPLES, device)
    torch.cuda.empty_cache()
    gan_feats = extract_features(inception, gan_imgs, device)
    gan_fid = compute_fid(real_feats, gan_feats)
    print(f"  gan fid: {gan_fid:.2f}")

    print(f"\n[diffusion] generating {N_SAMPLES} samples...")
    diff_imgs = generate_diffusion(N_SAMPLES, device)
    torch.cuda.empty_cache()
    diff_feats = extract_features(inception, diff_imgs, device)
    diff_fid = compute_fid(real_feats, diff_feats)
    print(f"  diffusion fid: {diff_fid:.2f}")

    print(f"\n[summary]")
    print(f"gan fid: {gan_fid:.2f}")
    print(f"diffusion fid: {diff_fid:.2f}")
    winner = "diffusion" if diff_fid < gan_fid else "gan"
    print(f"relative winner: {winner} (lower fid = closer to real distribution)")
    print(f"note: absolute fid values are inflated due to inception/medical domain mismatch, the relative gap is what matters for this comparison")

    plot_results(gan_fid, diff_fid)


if __name__ == "__main__":
    main()