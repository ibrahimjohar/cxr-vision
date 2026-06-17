# cxr-vision — project handoff

## identity
- **student:** ibrahim, cs @ fast nuces karachi
- **course:** deep learning for perception (cs4045)
- **repo:** github.com/ibrahimjohar/cxr-vision (public)
- **local path:** `C:\Ibrahim\Personal\University Stuff\Deep Learning for Perceptron\chest_xray_project`
- **purpose:** portfolio project for internship applications demonstrating breadth across classical cv, deep learning, and deployment

---

## hardware & environment
- gpu: gtx 1650 4gb vram
- os: windows, powershell
- python: 3.13
- venv at project root (`venv\Scripts\activate`)
- pytorch 2.7.1+cu118
- clip: installed via `pip install git+https://github.com/openai/CLIP.git`
- open_clip: installed via `pip install open_clip_torch`
- transformers: 4.45.0 (pinned — newer versions break on python 3.13 due to torchaudio/pyo3 issue)
- scipy: installed (needed for fid computation)

---

## dataset
- rsna pneumonia detection challenge (kaggle)
- 26,684 images total: 6,012 pneumonia (opacity), 20,672 normal
- format: dicom converted to png via `scripts/convert_dicom.py`
- location: `data/raw/rsna/stage_2_train_images_png/`
- labels csv: `data/raw/rsna/stage_2_train_labels.csv` (has bounding box coords for opacity cases)
- splits: train 21,347 / val 2,668 / test 2,669 (stratified 80/10/10)

---

## project structure
```
cxr-vision/
├── configs/config.py                        ✅ all hyperparameters
├── data/raw/rsna/                           ✅ pngs + labels csv
├── src/classical/preprocessing.py           ✅ clahe, gaussian, canny, sobel
├── src/models/classifier.py                 ✅ resnet18
├── src/models/vae.py                        ✅ conv vae
├── src/models/gan.py                        ✅ dcgan
├── src/models/attention_unet.py             ✅ attention u-net
├── src/models/diffusion.py                  ✅ ddpm u-net + ddim sampler
├── src/training/train_classifier.py         ✅ progressive unfreezing + early stopping
├── src/training/train_vae.py                ✅ fixed beta, recon-based checkpointing
├── src/training/train_gan.py                ✅ periodic checkpointing
├── src/training/train_unet.py               ✅ real bbox masks, dice checkpointing
├── src/training/train_diffusion.py          ✅ ddpm training + ddim sampling
├── src/utils/dataset.py                     ✅ classifier/vae/gan dataloader
├── src/utils/seg_dataset.py                 ✅ bbox→mask segmentation dataloader
├── src/evaluation/eval_vae.py               ✅ mse + ssim anomaly scoring
├── src/evaluation/eval_vae_ssim.py          ✅ ssim vs mse comparison
├── src/evaluation/eval_unet.py              ✅ split dice (opacity-only vs overall)
├── src/evaluation/clip_eval.py              ✅ openai clip + biomedclip comparison
├── src/evaluation/eval_generative.py        ✅ fid comparison gan vs diffusion
├── scripts/convert_dicom.py                 ✅ dicom→png conversion
├── main.py                                  ✅ entry point, --stage flag
├── api/main.py                              ❌ fastapi — not built
├── modal_app/serve.py                       ❌ modal serverless — not built
├── dashboard/                               ❌ next.js — not built
├── .github/workflows/deploy.yml             ❌ ci/cd — not built
├── Dockerfile                               ❌ not built
└── requirements.txt, README.md              ✅
```

---

## run commands
always activate venv first: `venv\Scripts\activate`
always run from project root via main.py:
```bash
python main.py --stage classical
python main.py --stage classifier
python main.py --stage vae
python main.py --stage gan
python main.py --stage unet
python main.py --stage diffusion
```
eval scripts run directly:
```bash
python src/evaluation/eval_vae.py
python src/evaluation/eval_unet.py
python src/evaluation/clip_eval.py
python src/evaluation/eval_generative.py
```

---

## config values (configs/config.py)

### general
```python
IMAGE_SIZE = 224
BATCH_SIZE = 16
SEED = 42
DEVICE = "cuda"
```

### classifier
```python
CNN_BACKBONE = "resnet18"
PRETRAINED = True
FREEZE_BACKBONE = True
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
SCHEDULER = "cosine"
```

### vae
```python
VAE_LATENT_DIM = 128
VAE_EPOCHS = 20
VAE_LR = 1e-3
VAE_BETA = 0.1
VAE_KL_ANNEAL = 1       # annealing disabled — fixed beta from epoch 1
VAE_PATIENCE = 5
```

### gan
```python
GAN_LATENT_DIM = 100
GAN_EPOCHS = 20
GAN_LR_G = 2e-4
GAN_LR_D = 2e-4
GAN_BETA1 = 0.5
```

### u-net
```python
UNET_FEATURES = [32, 64, 128, 256]   # reduced from [64,128,256,512] for 4gb vram
UNET_EPOCHS = 20
UNET_LR = 1e-4
```

### diffusion
```python
DIFFUSION_IMG_SIZE = 128       # lower than 224 to fit 4gb vram
DIFFUSION_TIMESTEPS = 1000
DIFFUSION_BASE_CH = 64
DIFFUSION_EPOCHS = 10
DIFFUSION_LR = 2e-4
DIFFUSION_BATCH = 8
DIFFUSION_DDIM_STEPS = 50
```

---

## model results — all numerical evaluations

### 1. resnet18 classifier
- architecture: resnet18 backbone (imagenet pretrained) → dropout → linear(512→256) → relu → dropout → linear(256→2)
- training: frozen backbone first 5 epochs, progressive unfreeze at 0.1× lr, early stopping patience=5
- **best val accuracy: 85.08%** (epoch 11, early stopped epoch 16)
- train/val split: 21,347 / 2,668
- checkpoints: `outputs/checkpoints/best_resnet18.pth`

### 2. vae anomaly detection
- architecture: conv encoder (4 conv layers, 224→14, flatten→fc_mu/fc_logvar) + conv decoder
- params: 20,696,001
- training: normal-only (16,537 images), fixed beta=0.1, 20 epochs
- best val recon loss: 0.0094 (epoch 18)
- kl at convergence: ~0.058 (healthy, nonzero)
- **anomaly detection auc (mse): 0.5105** — chance level
- **anomaly detection auc (ssim): 0.4829** — also chance
- separation: 1.037× (normal vs pneumonia recon error)
- verdict: documented negative result. vanilla vae reconstruction error fails to separate subtle localized cxr pathology. consistent with literature (~0.55–0.62 on easier benchmarks). method limitation, not implementation bug.
- checkpoint: `outputs/checkpoints/best_vae.pth`

### 3. dcgan
- architecture: generator (linear→512×7×7→convtranspose to 224, tanh, batchnorm) + discriminator (strided conv, leakyrelu 0.2, sigmoid)
- generator params: 5,320,640 | discriminator params: 693,632
- training: all 21,347 images, 20 epochs, no early stopping
- training dynamics: d(x) stabilized at 0.69–0.75, d(g(z)) at 0.25–0.31 (healthy equilibrium)
- **best checkpoint: epoch 10** (gan_epoch010.pth) — sharpest images, best d/g balance
- epoch 20 shows discriminator gaining edge (d(x)=0.85, d(g(z))=0.15) — past peak
- **fid vs real: 89.56** (500 samples, inception features, relative comparison only)
- visual result: recognizable chest x-rays from epoch 5 onward, diverse lung fields, ribcage, spine, diaphragm
- checkpoint: `outputs/checkpoints/gan_epoch010.pth`

### 4. attention u-net segmentation
- architecture: 4-level encoder/decoder with attention gates, features=[32,64,128,256]
- params: 7,851,197
- supervision: real rsna bounding boxes converted to binary masks, scaled 1024→224
- training: all images (opacity + normal), combined dice+bce loss, 20 epochs
- **best val dice (overall): 0.7630** — inflated by 2,068 all-black normal masks
- **opacity-only dice: 0.4036** (n=601) ← honest localization quality
- **normal-only dice: 0.8700** (n=2,068) ← false-positive suppression
- failure mode: model learns lung-region prior (predicts both lung fields) rather than precise opacity boundaries — known weak-supervision limitation with coarse box targets
- checkpoint: `outputs/checkpoints/best_unet.pth` (epoch 16)

### 5. ddpm diffusion model
- architecture: time-conditioned resnet-block u-net, no self-attention, 3-level encoder/decoder
- params: 6,613,761
- training: all 21,347 images at 128×128, 10 epochs, linear noise schedule (β: 1e-4→0.02)
- loss (noise mse): 0.0216 (epoch 1) → 0.0118 (epoch 10), converged by epoch 7
- sampling: ddim, 50 steps (vs 1000 training steps) — enables cpu inference
- **fid vs real: 64.09** (500 samples, relative comparison)
- **relative winner over gan: ~25 fid points lower** (64.09 vs 89.56)
- visual quality: recognizable thoracic anatomy from epoch 1, sharp ribcage/lung fields by epoch 10
- checkpoint: `outputs/checkpoints/diffusion_epoch010.pth`
- note: trained at 128×128 (vs gan 224×224) due to vram constraints

### 6. clip zero-shot evaluation
#### openai clip (vit-b/32)
- training data: 400m natural image-text pairs (no medical images)
- plain prompts: **auc=0.5093, acc=0.2252**
- clinical prompts: auc=0.4400, acc=0.7744 (high acc = majority class bias, not skill)
- descriptive prompts: auc=0.3854, acc=0.2349
- verdict: chance-level across all prompt formulations. domain mismatch.

#### biomedclip (pubmedbert-256 vit-b/16)
- training data: 15m biomedical figure-caption pairs from pubmed central
- **auc=0.8386, acc=0.6875**
- **delta vs openai clip: +0.3293 auc**
- verdict: near-supervised-cnn performance (85%) with zero task-specific training

#### key finding
the +0.33 auc gap isolates domain-specific pretraining effect. accuracy/auc trap demonstrated: clinical prompts gave 77% accuracy but 0.44 auc — model defaulted to majority class (normal), not real detection.

---

## generative comparison summary

| model | fid | sampling | training stability | resolution |
|-------|-----|----------|--------------------|-----------|
| dcgan | 89.56 | 1 forward pass | requires d/g balance | 224×224 |
| ddpm | 64.09 | 50 ddim steps | stable mse loss | 128×128 |

---

## key resolved issues (windows/environment)

- **num_workers=0** in all dataloaders — windows multiprocessing crash
- **sys.path** via `os.getcwd()` — run all scripts from project root via main.py
- **gitignore encoding** — powershell `echo >` writes utf-16 with bom, git can't read it. always edit .gitignore in vscode as utf-8. fix: `python -c "open('.gitignore','w',encoding='utf-8').write(...)"`
- **.pth files in git history** — `git rm --cached` + `git filter-branch` + `git gc --prune=now --aggressive` to clean. current .gitignore has `*.pth` but tracked files need manual untracking.
- **transformers version** — pin to 4.45.0. newer versions import torchaudio which has a broken windows dll on this machine (winError 127).
- **unet_features=[32,64,128,256]** — reduced from [64,128,256,512] for 4gb vram. do not change.
- **diffusion batch=8 at 128×128** — tightest memory config. do not increase without testing oom first.

---

## code style (apply to all code)
- lowercase comments
- no `# ────` section dividers
- no artificial line spacing
- comments only where genuinely needed
- code should look human-written, not llm-generated

---

## completed work
- ✅ all 7 model files built and understood
- ✅ all 5 training scripts with proper techniques (early stopping, progressive unfreezing, etc.)
- ✅ all evaluation scripts with honest split metrics
- ✅ classifier, vae, gan, unet, diffusion trained on real rsna data
- ✅ clip + biomedclip zero-shot evaluated
- ✅ fid comparison gan vs diffusion
- ✅ gitignore encoding fixed (utf-8)
- ✅ pycache untracked from git
- ✅ repo public on github

---

## pending — deployment phase

### dashboard plan (next.js on vercel)
8 pages total:

**static result pages (pre-computed json + png):**
1. overview — pipeline summary, dataset stats, model results table
2. classical preprocessing — raw vs clahe vs canny vs sobel side-by-side
3. cnn classifier — training curve, 85% accuracy, confusion matrix
4. vae anomaly detection — reconstruction figures, separation histogram, honest negative result
5. attention u-net — split dice results, prediction figures (input/gt/prediction)
6. clip zero-shot — openai vs biomedclip bar chart, accuracy/auc trap explanation

**interactive/live pages:**
7. gan vs diffusion — epoch progression grids for both, fid comparison chart, "generate" button that calls modal endpoint to generate a fresh sample from either model (user picks which)
8. live inference — upload x-ray → runs through:
   - resnet18 classifier (normal/pneumonia + confidence bar)
   - attention u-net (opacity heatmap overlay)
   - vae (reconstruction side-by-side + recon error score)
   - openai clip + biomedclip (similarity scores side-by-side)

### deployment stack
- fastapi (`api/main.py`) — inference endpoints
- modal (`modal_app/serve.py`) — serverless deployment, free tier, cpu-only, hard $0 spend cap
- vercel — next.js dashboard hosting
- docker — containerize fastapi app
- github actions (`.github/workflows/deploy.yml`) — auto-deploy to modal on push to main

### modal endpoints needed
- `POST /predict` — classifier + unet + vae on uploaded image
- `POST /clip` — openai clip + biomedclip scores on uploaded image  
- `POST /generate/gan` — generate one gan sample
- `POST /generate/diffusion` — generate one ddim sample (50 steps on cpu, ~10-20s)

### results export script needed
- `scripts/export_results.py` — saves all pre-computed metrics, figures as static json + png into `dashboard/public/results/`
- this feeds all static dashboard pages without hitting modal at runtime

---

## deployment story (for interviews)
"trained locally on gtx 1650 → dockerized → ci/cd via github actions → modal serverless (cpu inference) → vercel next.js dashboard with live inference"

---

## honest result framing (for writeup/interviews)

**classifier:** resnet18 with progressive unfreezing achieves 85% val accuracy on rsna binary classification.

**vae:** properly-trained vanilla vae (latent 128, β=0.1, healthy kl ~0.058) achieves chance-level anomaly detection (auc 0.51) on rsna. global reconstruction error fails to capture localized opacities. method limitation, not implementation failure. motivates supervised approaches.

**gan:** dcgan produces visually realistic synthetic chest x-rays by epoch 5, stable adversarial equilibrium throughout 20 epochs. best checkpoint at epoch 10 (fid 89.56).

**diffusion:** ddpm with ddim sampling (50 steps) outperforms dcgan in fid (64.09 vs 89.56) and visual quality. demonstrates score-based generative modeling as an alternative to adversarial training. trained at 128×128 due to vram constraints.

**u-net:** attention u-net achieves 0.40 dice on opacity-containing images (honest localization metric) and 0.87 on normal images. overall 0.76 dice is inflated by all-black normal masks — reported separately. learns lung-region localization under coarse box supervision.

**clip:** general clip (auc 0.51) vs domain-specific biomedclip (auc 0.84) — both zero-shot, no rsna training. +0.33 auc gap isolates domain pretraining effect. demonstrates vision-language model domain transfer to medical imaging.

---

## session behavior note
when context window reaches ~90% full, generate updated handoff for download before continuing. this file should always reflect current project state.