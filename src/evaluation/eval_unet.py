import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from tqdm import tqdm

from configs.config import CHECKPOINTS_DIR, DEVICE
from src.models.attention_unet import AttentionUNet
from configs.config import UNET_FEATURES
from src.utils.seg_dataset import get_seg_dataloaders

# honest u-net evaluation. the training dice averages in the all-black normal masks,
# which score ~1.0 each and inflate the number. here we report dice split three ways:
#   overall    - every test image (comparable to the inflated training metric)
#   opacity    - only images that actually contain a box (the real localization quality)
#   normal     - only normal images (measures false-positive suppression)
# the opacity-only number is the one that reflects whether the model localizes pneumonia.


def dice_score(pred, target, smooth=1.0, thresh=0.5):
    pred = (pred > thresh).float().view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return ((2 * inter + smooth) / (pred.sum() + target.sum() + smooth)).item()


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    _, _, test_loader = get_seg_dataloaders()

    model = AttentionUNet(in_channels=1, num_classes=1, features=UNET_FEATURES).to(device)
    ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, "best_unet.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[model] loaded unet from epoch {ckpt['epoch']}")

    overall, opacity, normal = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = model(images)
            for i in range(images.size(0)):
                d = dice_score(preds[i], masks[i])
                overall.append(d)
                if masks[i].sum() > 0:
                    opacity.append(d)
                else:
                    normal.append(d)

    print(f"\n[results] (n_test={len(overall)})")
    print(f"overall dice: {np.mean(overall):.4f} (n={len(overall)})")
    print(f"opacity-only dice: {np.mean(opacity):.4f} (n={len(opacity)}) <- real localization quality")
    print(f"normal-only dice: {np.mean(normal):.4f} (n={len(normal)}) <- false-positive suppression")
    print(f"\nthe overall number is inflated by the {len(normal)} all-black normal masks.")
    print(f"opacity-only dice is the honest measure of localization performance.")


if __name__ == "__main__":
    main()