"""
main.py — Project entry point

Runs each stage of the pipeline in order:
    Stage 1: Classical preprocessing (OpenCV)
    Stage 2: CNN classifier training
    Stage 3: VAE training
    Stage 4: GAN training
    Stage 5: Attention U-Net training

Run individual stages with:
    python main.py --stage classical
    python main.py --stage classifier
    python main.py --stage vae
    python main.py --stage gan
    python main.py --stage unet
    python main.py --stage all
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_classical():
    print("\n" + "="*60)
    print("STAGE 1: Classical Preprocessing (OpenCV)")
    print("="*60)
    from src.classical.preprocessing import preprocess_xray
    from configs.config import DATA_RAW
    import numpy as np
    from PIL import Image

    # Create a test image if none exist
    dummy = os.path.join(DATA_RAW, "dummy", "dummy_0000.png")
    if os.path.exists(dummy):
        results = preprocess_xray(dummy)
        print("Classical preprocessing passed.")
        for k, v in results.items():
            print(f"  {k:15s}: {v.shape}")
    else:
        print("[INFO] No images found. Run dataset download first.")


def run_classifier():
    print("\n" + "="*60)
    print("STAGE 2: CNN Classifier Training")
    print("="*60)
    from src.training.train_classifier import train
    train(backbone="resnet18")


def run_vae():
    from src.training.train_vae import train
    train()


def run_gan():
    print("\n" + "="*60)
    print("STAGE 4: DCGAN")
    print("="*60)
    import torch
    from src.models.gan import build_gan, generator_loss, discriminator_loss
    G, D = build_gan()
    z     = torch.randn(4, 100)
    fake  = G(z)
    score = D(fake)
    print(f"Generator output: {fake.shape}")
    print(f"Discriminator scores: {score.squeeze().tolist()}")
    print("To train: implement src/training/train_gan.py (next step)")


def run_unet():
    print("\n" + "="*60)
    print("STAGE 5: Attention U-Net")
    print("="*60)
    import torch
    from src.models.attention_unet import AttentionUNet
    model = AttentionUNet(in_channels=1, num_classes=1)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print("To train: implement src/training/train_unet.py (next step)")


STAGES = {
    "classical":  run_classical,
    "classifier": run_classifier,
    "vae":        run_vae,
    "gan":        run_gan,
    "unet":       run_unet,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chest X-ray Pipeline")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="all",
        help="Which stage to run"
    )
    args = parser.parse_args()

    if args.stage == "all":
        for name, fn in STAGES.items():
            fn()
    else:
        STAGES[args.stage]()
