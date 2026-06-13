import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import CHECKPOINTS_DIR, FIGURES_DIR, IMAGE_SIZE, BATCH_SIZE, DEVICE
from src.models.vae import VAE
from src.utils.dataset import build_rsna_dataset, ChestXrayDataset

# re-scores the EXISTING vae checkpoint with two anomaly metrics and compares them:
#   1. mse - per-pixel squared error (what we used before)
#   2. 1-ssim - structural dissimilarity, sensitive to local texture/structure changes
# pneumonia is a structural/textural change, so ssim-based error should separate
# the classes better than mse if the model is reconstructing at all usefully.
# no retraining - this only changes how we measure reconstruction error.


def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])


def get_test_loader(data_dir=None):
    _, _, test_p, _, _, test_l = build_rsna_dataset(data_dir)
    ds = ChestXrayDataset(test_p, test_l, transform=get_transforms())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


def compute_scores(model, loader, device):
    model.eval()
    mse_scores, ssim_scores, labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="scoring"):
            images = batch["image"].to(device)
            recon, _, _ = model(images)

            #mse per image
            err = ((recon - images) ** 2).mean(dim=[1, 2, 3])
            mse_scores.extend(err.cpu().numpy())

            #ssim per image - computed on cpu numpy, 1-ssim so higher = more anomalous
            imgs_np = images.cpu().squeeze(1).numpy()
            recon_np = recon.cpu().squeeze(1).numpy()
            for i in range(imgs_np.shape[0]):
                s = ssim(imgs_np[i], recon_np[i], data_range=1.0)
                ssim_scores.append(1.0 - s)

            labels.extend(batch["label"].numpy())
    return np.array(mse_scores), np.array(ssim_scores), np.array(labels)


def report(name, scores, labels):
    auc = roc_auc_score(labels, scores)
    nmean = scores[labels == 0].mean()
    pmean = scores[labels == 1].mean()
    print(f"\n[{name}]")
    print(f"normal mean: {nmean:.5f}")
    print(f"pneumonia mean: {pmean:.5f}")
    print(f"separation: {pmean/nmean:.3f}x")
    print(f"auc-roc: {auc:.4f}")
    return auc


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model = VAE().to(device)
    ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "best_vae.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[model] loaded vae from epoch {ckpt['epoch']}")

    test_loader = get_test_loader()
    mse_scores, ssim_scores, labels = compute_scores(model, test_loader, device)

    mse_auc = report("mse", mse_scores, labels)
    ssim_auc = report("1-ssim", ssim_scores, labels)

    print(f"\n[verdict] mse auc={mse_auc:.4f} | ssim auc={ssim_auc:.4f} | delta={ssim_auc-mse_auc:+.4f}")


if __name__ == "__main__":
    main()