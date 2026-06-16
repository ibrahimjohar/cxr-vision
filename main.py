"""
main.py - project entry point

Runs each stage of the pipeline:
    classical  : Classical preprocessing (OpenCV)
    classifier : CNN classifier training
    vae        : VAE training
    gan        : GAN training
    unet       : Attention U-Net training
    diffusion  : DDPM diffusion training

Run individual stages with:
    python main.py --stage classical
    python main.py --stage classifier
    python main.py --stage vae
    python main.py --stage gan
    python main.py --stage unet
    python main.py --stage diffusion
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
    from src.training.train_gan import train
    train()


def run_unet():
    from src.training.train_unet import train
    train()


def run_diffusion():
    from src.training.train_diffusion import train
    train()


STAGES = {
    "classical": run_classical,
    "classifier": run_classifier,
    "vae": run_vae,
    "gan": run_gan,
    "unet": run_unet,
    "diffusion": run_diffusion,
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