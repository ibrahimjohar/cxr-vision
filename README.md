# Chest X-ray Enhancement & Segmentation Pipeline

A progressively built medical imaging pipeline implementing course concepts end-to-end.

## Project Structure

```
chest_xray_project/
│
├── configs/
│   └── config.py              ← all hyperparameters live here
│
├── data/
│   ├── raw/                   ← downloaded datasets go here
│   ├── processed/             ← OpenCV preprocessed images
│   └── augmented/             ← GAN-generated synthetic images
│
├── src/
│   ├── classical/
│   │   └── preprocessing.py   ← OpenCV: CLAHE, Gaussian, Canny, Sobel
│   │
│   ├── models/
│   │   ├── classifier.py      ← ResNet transfer learning classifier
│   │   ├── vae.py             ← Variational Autoencoder
│   │   ├── gan.py             ← DCGAN generator + discriminator
│   │   └── attention_unet.py  ← U-Net with attention gates
│   │
│   ├── training/
│   │   └── train_classifier.py← full training loop with checkpointing
│   │
│   ├── evaluation/            ← metrics, CLIP evaluation (coming)
│   └── utils/
│       └── dataset.py         ← DataLoader factory, transforms
│
├── notebooks/                 ← Jupyter notebooks for exploration
├── outputs/
│   ├── checkpoints/           ← saved model weights
│   ├── logs/                  ← training logs
│   └── figures/               ← plots and visualizations
│
├── main.py                    ← entry point
├── requirements.txt
└── README.md
```

## Pipeline Stages

| Stage | What | Course Week |
|-------|------|-------------|
| 1. Classical | CLAHE + Gaussian + Canny edge detection | Wk 1-2 |
| 2. CNN Classifier | ResNet fine-tuned on X-rays | Wk 5-6 |
| 3. VAE | Latent space + anomaly detection | Wk 7 |
| 4. DCGAN | Synthetic X-ray generation | Wk 8 |
| 5. Attention U-Net | Segmentation with attention gates | Wk 10-11 |
| 6. CLIP Evaluation | Zero-shot pathology alignment | Wk 13 |

## Setup

### 1. Install dependencies

```bash
# GPU (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision torchaudio

# Project dependencies
pip install -r requirements.txt
```

### 2. Download Dataset (RSNA Pneumonia — free, small, well-labeled)

```bash
# Install Kaggle CLI
pip install kaggle

# Place kaggle.json in ~/.kaggle/
# Download dataset
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip -d data/raw/rsna/
```

### 3. Convert DICOM to PNG (RSNA provides .dcm files)

```bash
python scripts/convert_dicom.py   # (to be added)
```

### 4. Run the pipeline

```bash
# Individual stages
python main.py --stage classical
python main.py --stage classifier
python main.py --stage vae
python main.py --stage gan
python main.py --stage unet

# Full pipeline
python main.py --stage all
```

## Research Framing

**Title:** Classical vs Deep Learning Approaches to Chest X-ray Enhancement and Pathology Detection

**Contributions:**
1. Comparison of classical (OpenCV) vs deep preprocessing on X-ray quality
2. Transfer learning with attention-augmented segmentation
3. VAE-based anomaly detection using reconstruction error
4. DCGAN for synthetic X-ray augmentation on limited labeled data
5. CLIP zero-shot evaluation as a label-free evaluation metric
