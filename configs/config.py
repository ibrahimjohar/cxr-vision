"""
Central configuration for the Chest X-ray Enhancement & Segmentation Pipeline.
Edit values here — everything else reads from this file.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
DATA_AUGMENTED  = os.path.join(BASE_DIR, "data", "augmented")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "outputs", "checkpoints")
LOGS_DIR        = os.path.join(BASE_DIR, "outputs", "logs")
FIGURES_DIR     = os.path.join(BASE_DIR, "outputs", "figures")

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME    = "rsna-pneumonia"       # swap to "chestxray14" when available
IMAGE_SIZE      = 224                    # resize all images to this
NUM_CLASSES     = 2                      # binary: normal vs pneumonia
TRAIN_SPLIT     = 0.8
VAL_SPLIT       = 0.1
TEST_SPLIT      = 0.1

# ─── Classical Preprocessing (OpenCV stage) ───────────────────────────────────
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA  = 1.0
CLAHE_CLIP      = 2.0                   # contrast limiting for CLAHE
CLAHE_GRID      = (8, 8)
CANNY_LOW       = 50
CANNY_HIGH      = 150

# ─── CNN Classifier ───────────────────────────────────────────────────────────
CNN_BACKBONE    = "resnet18"            # resnet18 | resnet50 | densenet121
PRETRAINED      = True                  # use ImageNet weights
FREEZE_BACKBONE = True                 # fine-tune entire network

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 16
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4                  # L2 regularization
DROPOUT_RATE    = 0.3
SCHEDULER       = "cosine"             # cosine | step | none

# ─── VAE ──────────────────────────────────────────────────────────────────────
VAE_LATENT_DIM  = 128
VAE_EPOCHS      = 20
VAE_LR          = 1e-3
VAE_BETA        = 1.0                   # weight on KL term (1 = standard VAE)

# ─── GAN ──────────────────────────────────────────────────────────────────────
GAN_LATENT_DIM  = 100
GAN_EPOCHS      = 50
GAN_LR_G        = 2e-4
GAN_LR_D        = 2e-4
GAN_BETA1       = 0.5                   # Adam beta1 for GAN (standard DCGAN value)

# ─── Attention U-Net ──────────────────────────────────────────────────────────
UNET_FEATURES   = [32, 64, 128, 256]
UNET_EPOCHS     = 30
UNET_LR         = 1e-4

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE          = "cuda"                # will fallback to cpu if cuda unavailable

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED            = 42
