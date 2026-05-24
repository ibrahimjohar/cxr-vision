# cxr-vision

End-to-end chest X-ray pathology detection pipeline — classical OpenCV preprocessing through CNN, VAE, GAN, Attention U-Net, and CLIP evaluation in PyTorch.

---

## Overview

This project implements a full medical imaging pipeline for pneumonia detection on the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) dataset. It covers classical computer vision baselines through modern deep learning architectures, with a deployed dashboard for visualization.

**Stack:** Python · PyTorch · OpenCV · FastAPI · Modal · Next.js · Docker · GitHub Actions

---

## Pipeline

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. Preprocessing | CLAHE + Gaussian + Canny (OpenCV) | Enhance contrast, detect anatomical edges |
| 2. Classification | ResNet18 fine-tuned | Binary pneumonia detection |
| 3. Anomaly Detection | Variational Autoencoder | Reconstruction error as unsupervised anomaly score |
| 4. Augmentation | DCGAN | Synthetic X-ray generation to address class imbalance |
| 5. Segmentation | Attention U-Net | Pixel-level localization of pathological regions |
| 6. Evaluation | CLIP zero-shot | Label-free alignment between images and clinical text prompts |

---

## Project Structure

```
cxr-vision/
├── configs/config.py              ← all hyperparameters
├── data/
│   ├── raw/                       ← RSNA DICOM + converted PNGs
│   ├── processed/                 ← preprocessed outputs
│   └── augmented/                 ← GAN-generated samples
├── src/
│   ├── classical/preprocessing.py ← OpenCV pipeline
│   ├── models/
│   │   ├── classifier.py          ← ResNet transfer learning
│   │   ├── vae.py                 ← VAE + reparameterization trick
│   │   ├── gan.py                 ← DCGAN generator + discriminator
│   │   └── attention_unet.py      ← U-Net with attention gates
│   ├── training/                  ← training loops per model
│   ├── evaluation/                ← metrics + CLIP eval
│   └── utils/dataset.py           ← DataLoader factory
├── api/main.py                    ← FastAPI inference endpoint
├── modal_app/serve.py             ← Modal serverless wrapper
├── dashboard/                     ← Next.js + TypeScript visualization
├── .github/workflows/deploy.yml   ← CI/CD
├── Dockerfile
├── main.py                        ← pipeline entry point
└── requirements.txt
```

---

## Setup

**1. Install dependencies**
```bash
# GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

**2. Download dataset**
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip -d data/raw/rsna/
```

**3. Convert DICOM to PNG**
```bash
python scripts/convert_dicom.py
```

**4. Run pipeline**
```bash
python main.py --stage classical
python main.py --stage classifier
python main.py --stage vae
python main.py --stage gan
python main.py --stage unet
python main.py --stage all
```

---

## Deployment

- **Model serving:** FastAPI + Modal (serverless, free tier)
- **Dashboard:** Next.js deployed on Vercel
- **CI/CD:** GitHub Actions on push to `main`

---

## Research Framing

**Problem:** Automating chest X-ray diagnosis faces two core challenges — limited labeled data and difficulty localizing subtle pathological regions.

**Approach:**
1. Classical preprocessing establishes an interpretable baseline
2. Transfer learning classifier provides a strong discriminative benchmark
3. VAE trained on normal X-rays uses reconstruction error as an unsupervised anomaly score
4. DCGAN generates synthetic pneumonia samples to address class imbalance
5. Attention U-Net segments pathological regions without requiring dense annotations
6. CLIP validates findings via zero-shot clinical text alignment — no labels required

---

## Dataset

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) — ~26,000 chest X-rays, binary labeled (Normal / Pneumonia).