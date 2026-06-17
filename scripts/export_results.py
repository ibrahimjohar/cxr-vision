import os
import sys
import json
import shutil
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())

from configs.config import *

OUT_DIR = "dashboard/public/results"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

FIGS = "outputs/figures"


def save_json(data, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {filename}")


def copy_fig(src, dst_name):
    src_path = os.path.join(FIGS, src)
    if not os.path.exists(src_path):
        print(f"  missing: {src_path}")
        return None
    dst_path = os.path.join(IMG_DIR, dst_name)
    shutil.copy(src_path, dst_path)
    return f"images/{dst_name}"


def export_overview():
    save_json({
        "dataset": {
            "total": 26684,
            "pneumonia": 6012,
            "normal": 20672,
            "train": 21347,
            "val": 2668,
            "test": 2669
        },
        "models": [
            {"name": "ResNet18 Classifier", "metric": "Val Accuracy", "value": 85.08, "unit": "%"},
            {"name": "VAE Anomaly Detection", "metric": "AUC (MSE)", "value": 0.5105, "unit": ""},
            {"name": "DCGAN", "metric": "FID", "value": 89.56, "unit": ""},
            {"name": "DDPM Diffusion", "metric": "FID", "value": 64.09, "unit": ""},
            {"name": "Attention U-Net", "metric": "Dice (opacity-only)", "value": 0.4036, "unit": ""},
            {"name": "BiomedCLIP Zero-Shot", "metric": "AUC", "value": 0.8386, "unit": ""}
        ]
    }, "overview.json")


def export_preprocessing():
    import cv2

    labels_path = "data/raw/rsna/stage_2_train_labels.csv"
    img_dir = "data/raw/rsna/stage_2_train_images_png"

    sample_path = None
    with open(labels_path) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if parts[-1] == "1":
                pid = parts[0]
                candidate = os.path.join(img_dir, pid + ".png")
                if os.path.exists(candidate):
                    sample_path = candidate
                    break

    if sample_path is None:
        print("warning: no sample image found for preprocessing")
        save_json({"images": {}}, "preprocessing.json")
        return

    img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    canny_img = cv2.Canny(clahe_img, 50, 150)
    sobelx = cv2.Sobel(clahe_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(clahe_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_img = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))

    cv2.imwrite(os.path.join(IMG_DIR, "preprocess_raw.png"), img)
    cv2.imwrite(os.path.join(IMG_DIR, "preprocess_clahe.png"), clahe_img)
    cv2.imwrite(os.path.join(IMG_DIR, "preprocess_canny.png"), canny_img)
    cv2.imwrite(os.path.join(IMG_DIR, "preprocess_sobel.png"), sobel_img)

    save_json({
        "images": {
            "raw": "images/preprocess_raw.png",
            "clahe": "images/preprocess_clahe.png",
            "canny": "images/preprocess_canny.png",
            "sobel": "images/preprocess_sobel.png"
        }
    }, "preprocessing.json")


def export_classifier():
    ckpt_path = "outputs/checkpoints/best_resnet18.pth"
    history = None
    confusion = None

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        history = ckpt.get("history", None)
        confusion = ckpt.get("confusion_matrix", None)
        if confusion is not None and not isinstance(confusion, list):
            confusion = confusion.tolist()
    else:
        print("warning: classifier checkpoint not found, using known results")

    save_json({
        "final_val_accuracy": 85.08,
        "best_epoch": 11,
        "stopped_epoch": 16,
        "history": history,
        "confusion_matrix": confusion
    }, "classifier.json")


def export_vae():
    recon_imgs = []
    for epoch in [1, 5, 10, 15, 20]:
        path = copy_fig(f"vae_recon_epoch{epoch:03d}.png", f"vae_recon_epoch{epoch:03d}.png")
        if path:
            recon_imgs.append({"epoch": epoch, "path": path})

    anomaly_sep = copy_fig("vae_anomaly_separation.png", "vae_anomaly_separation.png")

    save_json({
        "auc_mse": 0.5105,
        "auc_ssim": 0.4829,
        "best_val_recon_loss": 0.0094,
        "kl_at_convergence": 0.058,
        "separation_ratio": 1.037,
        "recon_samples": recon_imgs,
        "anomaly_separation_figure": anomaly_sep
    }, "vae.json")


def export_unet():
    mask_imgs = []
    for epoch in [1, 5, 10, 15, 20]:
        path = copy_fig(f"unet_masks_epoch{epoch:03d}.png", f"unet_masks_epoch{epoch:03d}.png")
        if path:
            mask_imgs.append({"epoch": epoch, "path": path})

    save_json({
        "dice_overall": 0.7630,
        "dice_opacity_only": 0.4036,
        "dice_normal_only": 0.8700,
        "best_epoch": 16,
        "mask_samples": mask_imgs
    }, "unet.json")


def export_clip():
    clip_comparison = copy_fig("clip_comparison.png", "clip_comparison.png")
    clip_zeroshot = copy_fig("clip_zeroshot_prompts.png", "clip_zeroshot_prompts.png")

    save_json({
        "openai_clip": {
            "model": "ViT-B/32",
            "training_data": "400M natural image-text pairs",
            "results": [
                {"prompt_type": "plain", "auc": 0.5093, "accuracy": 0.2252},
                {"prompt_type": "clinical", "auc": 0.4400, "accuracy": 0.7744},
                {"prompt_type": "descriptive", "auc": 0.3854, "accuracy": 0.2349}
            ]
        },
        "biomedclip": {
            "model": "PubMedBERT-256 ViT-B/16",
            "training_data": "15M biomedical figure-caption pairs",
            "results": [
                {"prompt_type": "medical", "auc": 0.8386, "accuracy": 0.6875}
            ]
        },
        "delta_auc": 0.3293,
        "key_finding": "accuracy/auc trap: clinical prompts gave 77% accuracy but 0.44 AUC — model defaulted to majority class",
        "figures": {
            "comparison": clip_comparison,
            "zeroshot_prompts": clip_zeroshot
        }
    }, "clip.json")


def export_generative():
    gan_samples = []
    for epoch in [1, 5, 10, 15, 20]:
        path = copy_fig(f"gan_samples_epoch{epoch:03d}.png", f"gan_samples_epoch{epoch:03d}.png")
        if path:
            gan_samples.append({"epoch": epoch, "path": path})

    diff_samples = []
    for epoch in [1, 5, 10]:
        path = copy_fig(f"diffusion_samples_epoch{epoch:03d}.png", f"diffusion_samples_epoch{epoch:03d}.png")
        if path:
            diff_samples.append({"epoch": epoch, "path": path})

    fid_comparison = copy_fig("generative_fid_comparison.png", "generative_fid_comparison.png")

    save_json({
        "gan": {
            "fid": 89.56,
            "best_epoch": 10,
            "architecture": "DCGAN",
            "resolution": "224x224",
            "epoch_samples": gan_samples
        },
        "diffusion": {
            "fid": 64.09,
            "sampling": "DDIM 50 steps",
            "architecture": "DDPM U-Net",
            "resolution": "128x128",
            "epoch_samples": diff_samples
        },
        "fid_lower_is_better": True,
        "fid_comparison_figure": fid_comparison
    }, "generative.json")


if __name__ == "__main__":
    print("exporting static results...")
    export_overview()
    export_preprocessing()
    export_classifier()
    export_vae()
    export_unet()
    export_clip()
    export_generative()
    print(f"\ndone. results in {OUT_DIR}/")