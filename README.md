<img width="1560" height="886" alt="image" src="https://github.com/user-attachments/assets/173845ce-5ead-4f3a-aff2-aeb70a001522" />

# cxr-vision

End-to-end chest X-ray pathology detection pipeline — classical OpenCV preprocessing through CNN classification, VAE anomaly detection, generative modeling (GAN + diffusion), attention-based segmentation, and CLIP zero-shot evaluation, deployed as a live interactive dashboard.

Built for Deep Learning for Perception (CS4045) at FAST NUCES Karachi.

---

## Live

- **Dashboard:** [cxr-vision.vercel.app](https://cxr-vision.vercel.app/)
- **API:** [ibrahimjoharfarooqi--cxr-vision-inference-fastapi-app.modal.run](https://ibrahimjoharfarooqi--cxr-vision-inference-fastapi-app.modal.run) ([interactive docs](https://ibrahimjoharfarooqi--cxr-vision-inference-fastapi-app.modal.run/docs))

The dashboard's Inference and Generative pages call the live API directly — upload a real chest X-ray and get real model output, not mocked data.

---

## Results

Six models, trained and evaluated on the same 26,684-image dataset, reported as they actually came out — including a documented negative result and a metric trap.

| Model | Metric | Result | Verdict |
|---|---|---|---|
| ResNet18 Classifier | Val Accuracy / AUC | 85.08% / 0.8804 | Strong baseline |
| VAE (Anomaly Detection) | AUC (MSE) | 0.5105 | Chance-level — documented limitation |
| DCGAN | FID | 89.56 | Stable adversarial equilibrium |
| DDPM Diffusion | FID | 64.09 | Outperforms GAN by ~25 points |
| Attention U-Net | Dice (opacity-only) | 0.4036 | Honest metric — overall Dice (0.76) is inflated |
| CLIP zero-shot (OpenAI vs BiomedCLIP) | AUC | 0.51 vs 0.84 | Domain pretraining is the deciding factor |

Full breakdowns, confusion matrices, ROC curves, and reconstruction/generation samples for each model are on the live dashboard.

---

## Pipeline

| Stage | Method | Purpose |
|---|---|---|
| 1. Preprocessing | CLAHE + Gaussian + Canny (OpenCV) | Enhance contrast, detect anatomical edges |
| 2. Classification | ResNet18, fine-tuned | Binary pneumonia detection |
| 3. Anomaly Detection | Variational Autoencoder | Reconstruction error as unsupervised anomaly score |
| 4. Generative Modeling | DCGAN | Adversarial image synthesis, evaluated via FID |
| 5. Generative Modeling | DDPM (diffusion) | Score-based image synthesis via DDIM sampling, evaluated via FID |
| 6. Segmentation | Attention U-Net | Pixel-level localization from bounding-box supervision |
| 7. Zero-Shot Evaluation | CLIP / BiomedCLIP | Label-free image-text alignment, no RSNA training |

---

## Honest Findings

- **The VAE is a documented negative result, not a bug.** A properly trained VAE (healthy nonzero KL divergence, low reconstruction loss) still achieves chance-level AUC for anomaly detection. Global pixel-level reconstruction error is too coarse to catch pathology confined to a small region of the lung field — consistent with known literature on this approach.
- **The U-Net's overall Dice score is misleading on its own.** 0.76 is inflated by 2,068 all-black normal masks scoring well just by predicting near-empty. The honest number is the opacity-only Dice (0.40), measured only on the 601 cases that actually contain a labeled opacity.
- **CLIP prompt wording exposes an accuracy/AUC trap.** A "clinical" phrasing gives OpenAI CLIP 77% accuracy but an AUC of 0.44 — below chance. The model defaults to predicting the majority class ("normal"), and accuracy alone hides that; AUC exposes it.
- **Domain-specific pretraining is the single largest lever in the CLIP comparison.** Generic CLIP (AUC 0.51) and BiomedCLIP (AUC 0.84) share the same zero-shot pipeline — no RSNA training for either. The +0.33 AUC gap comes entirely from what each model was pretrained on.
- **Diffusion beats GAN on FID, but not on a fully matched comparison.** DDPM trained at 128×128 for 10 epochs; DCGAN trained at 224×224 for 20. Lower resolution tends to lower FID somewhat on its own, so the 25-point gap likely overstates the architectural advantage — though the qualitative difference in sample quality holds up independent of that caveat.

---

## Project Structure

```
cxr-vision/
├── configs/config.py              ← all hyperparameters
├── data/raw/                      ← RSNA DICOM + converted PNGs (gitignored)
├── outputs/checkpoints/           ← trained model weights (gitignored, ~1.08GB)
├── src/
│   ├── classical/preprocessing.py ← OpenCV pipeline
│   ├── models/
│   │   ├── classifier.py          ← ResNet18 transfer learning
│   │   ├── vae.py                 ← VAE + reparameterization trick
│   │   ├── gan.py                 ← DCGAN generator + discriminator
│   │   ├── attention_unet.py      ← U-Net with attention gates
│   │   └── diffusion.py           ← DDPM U-Net + DDIM sampler
│   ├── training/                  ← training loops per model
│   ├── evaluation/                ← metrics, confusion matrices, CLIP eval
│   └── utils/dataset.py           ← DataLoader factory
├── api/main.py                    ← FastAPI inference server (4 endpoints, rate limited)
├── modal_app/serve.py             ← Modal serverless deployment
├── dashboard/                     ← Next.js 16 + TypeScript visualization (9 pages)
├── .github/workflows/deploy.yml   ← CI/CD — auto-deploys api/ changes to Modal
└── main.py                        ← local pipeline entry point
```

Trained checkpoints are gitignored due to size and backed up separately at a private HuggingFace model repo, restored automatically during CI/CD deploys.

---

## Stack

**ML:** PyTorch 2.7, torchvision, OpenAI CLIP, BiomedCLIP (via `open_clip` + `transformers`)
**Backend:** FastAPI, slowapi (rate limiting), Modal (serverless CPU deployment)
**Frontend:** Next.js 16, React 19, Framer Motion, Recharts, Phosphor Icons
**Infra:** GitHub Actions (CI/CD), Vercel (dashboard hosting), HuggingFace Hub (checkpoint backup)

---

## Local Setup

**1. Install dependencies**
```bash
# GPU (CUDA 11.8) — trained locally on a GTX 1650, 4GB VRAM
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**2. Download the dataset**
```bash
pip install kaggle
# place kaggle.json in ~/.kaggle/
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip -d data/raw/rsna/
python scripts/convert_dicom.py
```

**3. Run training stages**
```bash
python main.py --stage classical
python main.py --stage classifier
python main.py --stage vae
python main.py --stage gan
python main.py --stage unet
python main.py --stage diffusion
```

**4. Run evaluation**
```bash
python src/evaluation/eval_classifier.py
python src/evaluation/eval_vae.py
python src/evaluation/eval_unet.py
python src/evaluation/clip_eval.py
python src/evaluation/eval_generative.py
```

**5. Run the API locally**
```bash
pip install slowapi
uvicorn api.main:app --reload
# interactive docs at http://127.0.0.1:8000/docs
```

**6. Run the dashboard locally**
```bash
cd dashboard
npm install
npm run dev
```

---

## Deployment Architecture

```
GTX 1650 (local training)
      │
      ▼
outputs/checkpoints/*.pth ──► HuggingFace Hub (private backup)
      │
      ▼
GitHub push to api/ src/ configs/ modal_app/
      │
      ▼
GitHub Actions: pulls checkpoints from HF, runs `modal deploy`
      │
      ▼
Modal (serverless CPU) ──► live FastAPI endpoint
      │
      ▼
Vercel (Next.js dashboard) ──► calls the live endpoint directly from the browser
```

- **API endpoints:** `POST /predict` (classifier + U-Net + VAE), `POST /clip` (CLIP + BiomedCLIP), `POST /generate/gan`, `POST /generate/diffusion`
- **Rate limiting:** tiered per endpoint by compute cost (5-15 requests/minute per IP), via `slowapi`
- **Cost control:** Modal workspace budget capped below the monthly free-credit allowance
- **CI/CD:** any push touching backend paths on `main` auto-redeploys; unrelated dashboard-only changes don't trigger a redeploy

---

## Dataset

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) — 26,684 chest X-rays, binary labeled (Normal / Pneumonia), stratified 80/10/10 train/val/test split.

---

## Roadmap

- Tighten CORS from `allow_origins=["*"]` to the Vercel domain specifically, now that it's known
- Upgrade `framer-motion` past its current React 18 peer-dependency declaration, to remove the `legacy-peer-deps` workaround in `dashboard/.npmrc`

**Note on Docker:** no Dockerfile is included. Both deployment targets build their own images natively — Modal from a `modal.Image` definition in `modal_app/serve.py`, Vercel from the Next.js source directly — so a separate container was never actually in the critical path. Serverless platforms handling their own builds was part of the reasoning for choosing them over a persistent-VM option like EC2.
