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
├── scripts/export_results.py                ✅ exports all static json + pngs to dashboard/public/results/
├── main.py                                  ✅ entry point, --stage flag
├── api/main.py                              ❌ fastapi — not built
├── modal_app/serve.py                       ❌ modal serverless — not built
├── dashboard/                               ✅ next.js initialized, foundation complete (see below)
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
export static results:
```bash
python scripts/export_results.py
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
- checkpoints: `outputs/checkpoints/best_resnet18.pth`

### 2. vae anomaly detection
- architecture: conv encoder (4 conv layers, 224→14, flatten→fc_mu/fc_logvar) + conv decoder
- params: 20,696,001
- training: normal-only (16,537 images), fixed beta=0.1, 20 epochs
- best val recon loss: 0.0094 (epoch 18), kl at convergence: ~0.058
- **anomaly detection auc (mse): 0.5105** — chance level
- **anomaly detection auc (ssim): 0.4829**
- verdict: documented negative result. method limitation, not implementation bug. consistent with literature (~0.55–0.62 vanilla, ~0.78 f-anogan).
- checkpoint: `outputs/checkpoints/best_vae.pth`

### 3. dcgan
- architecture: generator (linear→512×7×7→convtranspose to 224, tanh, batchnorm) + discriminator (strided conv, leakyrelu 0.2, sigmoid)
- generator params: 5,320,640 | discriminator params: 693,632
- training: all 21,347 images, 20 epochs
- d(x) stabilized at 0.69–0.75, d(g(z)) at 0.25–0.31 (healthy equilibrium)
- **best checkpoint: epoch 10** — sharpest images, best d/g balance
- **fid vs real: 89.56** (500 samples, relative comparison only)
- checkpoint: `outputs/checkpoints/gan_epoch010.pth`

### 4. attention u-net segmentation
- architecture: 4-level encoder/decoder with attention gates, features=[32,64,128,256]
- params: 7,851,197
- supervision: real rsna bounding boxes → binary masks, scaled 1024→224
- **best val dice (overall): 0.7630** — inflated by all-black normal masks
- **opacity-only dice: 0.4036** ← honest localization metric
- **normal-only dice: 0.8700**
- checkpoint: `outputs/checkpoints/best_unet.pth` (epoch 16)

### 5. ddpm diffusion model
- architecture: time-conditioned resnet-block u-net, 6,613,761 params
- training: 128×128, 10 epochs, linear noise schedule (β: 1e-4→0.02)
- sampling: ddim, 50 steps
- **fid vs real: 64.09** (~25 fid points better than dcgan)
- checkpoint: `outputs/checkpoints/diffusion_epoch010.pth`

### 6. clip zero-shot
- openai clip (vit-b/32): **auc=0.5093** — chance, domain mismatch
- biomedclip (pubmedbert-256 vit-b/16): **auc=0.8386**
- delta: +0.3293 auc isolates domain pretraining effect
- accuracy/auc trap: clinical prompts gave 77% acc but 0.44 auc — majority class exploitation

---

## generative comparison

| model | fid | sampling | resolution |
|-------|-----|----------|-----------|
| dcgan | 89.56 | 1 forward pass | 224×224 |
| ddpm  | 64.09 | 50 ddim steps | 128×128 |

---

## static results export — completed ✅
- `dashboard/public/results/` contains 7 json files + 33 images
- json files: overview, preprocessing, classifier, vae, unet, clip, generative
- images: all preprocessing pngs, vae recon epochs, unet mask epochs, gan/diffusion sample grids, fid comparison figure, clip figures

---

## key resolved issues

- **num_workers=0** in all dataloaders — windows multiprocessing crash
- **gitignore encoding** — always edit as utf-8 in vscode. fix: `python -c "open('.gitignore','w',encoding='utf-8').write(...)"`
- **.pth files in git history** — `git rm --cached` + `git filter-branch` + `git gc --prune=now --aggressive`. current .gitignore has `outputs/*` + `!outputs/figures/`
- **transformers version** — pin to 4.45.0
- **unet_features=[32,64,128,256]** — do not change for 4gb vram
- **diffusion batch=8 at 128×128** — tightest memory config

---

## dashboard — foundation complete ✅

### tech stack (dashboard/)
- next.js 16.2.9 (app router, typescript, tailwind v4)
- react 19.2.4
- framer-motion 11.3.0 (installed with --legacy-peer-deps)
- lenis 1.3.23 (smooth scroll)
- @phosphor-icons/react 2.1.10
- recharts 3.8.1

### critical environment fix
- `next.config.ts` — `allowedDevOrigins: ['192.168.56.1']` — required to unblock js execution

### completed files
- `app/globals.css` — theme tokens, type scale, interaction animations, orb keyframes, card hover
- `app/layout.tsx` — server component, metadata, font preloads, ClientProviders wrapper
- `components/ClientProviders.tsx` — lenis, theme state, view transitions api circular reveal
- `components/Navbar.tsx` — responsive, active indicator, theme toggle (mounted guard), mobile menu with sweep animation
- `components/CustomCursor.tsx` — exclusion blend mode, grows on hover, framer-motion spring
- `components/AnimatedBg.tsx` — three css-animated blurred orbs from corners, seamless loop

### design system
- **palette dark:** `#080510` base · `#0e0818` secondary · `#1a0a2e` tertiary · `#FFEDDF` text
- **palette light:** `#FFEDDF` base · `#f5e0cc` secondary · `#11001C` text
- **accent:** `#3A015C` → `#7B2FBE` gradient
- **fonts:** Instrument Serif (display) + Hanken Grotesk (ui/data)
- **theme transition:** view transitions api — circle expands from sun/moon button
  - desktop origin: `calc(100% - 320px) 32px`
  - mobile origin: `calc(100% - 56px) 32px`

### interactions
- custom cursor: exclusion blend, size 32px → 48px on hover
- buttons: scale(1.05) hover, scale(0.96) active
- nav links: sliding underline via `::after` pseudo-element
- cards: `.card-hover` — translateY(-4px) + border glow
- mobile menu: clip-path sweep down animation
- page enter: fade + translateY on `.page-wrapper > *`
- orbs: three blurred divs, corner-to-corner drift, seamless css keyframes

---

## pending — dashboard pages to build

### build order
1. `app/page.tsx` — home
2. `app/about/page.tsx` — about
3. `app/overview/page.tsx` — pipeline summary + results table
4. `app/preprocessing/page.tsx` — classical cv
5. `app/classifier/page.tsx` — resnet18
6. `app/vae/page.tsx` — anomaly detection
7. `app/unet/page.tsx` — segmentation
8. `app/clip/page.tsx` — zero-shot
9. `app/generative/page.tsx` — gan vs diffusion + live generate
10. `app/inference/page.tsx` — live upload

### page design rules (apply to every page)
- whileInView for all entrance animations (framer-motion, amount: 0.1)
- every interactive element has an animation
- data from json files in `dashboard/public/results/`
- charts built with recharts styled to theme (no default recharts colors)
- images from `dashboard/public/results/images/`
- glass cards for metric display
- instrument serif for headings, hanken grotesk for data
- fully responsive — desktop first, mobile handled separately
- no white-bg pngs — all charts built in-dashboard from json data

### data sources per page
- home/overview: `overview.json`
- preprocessing: `preprocessing.json` + image files
- classifier: `classifier.json` (history/confusion_matrix may be null)
- vae: `vae.json` + recon sample images
- unet: `unet.json` + mask sample images
- clip: `clip.json`
- generative: `generative.json` + epoch sample grids

---

## pending — deployment phase

### stack
- fastapi (`api/main.py`) — inference endpoints
- modal (`modal_app/serve.py`) — serverless, cpu-only, $0 spend cap
- vercel — dashboard hosting
- docker — containerize fastapi
- github actions — auto-deploy on push to main

### modal endpoints needed
- `POST /predict` — classifier + unet + vae on uploaded image
- `POST /clip` — openai clip + biomedclip scores
- `POST /generate/gan` — one gan sample
- `POST /generate/diffusion` — one ddim sample (50 steps cpu, ~10-20s)

---

## deployment story (interviews)
"trained locally on gtx 1650 → dockerized → ci/cd via github actions → modal serverless (cpu inference) → vercel next.js dashboard with live inference"

---

## honest result framing (interviews/writeup)

**classifier:** resnet18 with progressive unfreezing achieves 85% val accuracy on rsna binary classification.

**vae:** properly-trained vanilla vae achieves chance-level anomaly detection (auc 0.51). global reconstruction error fails on localized opacities. method limitation, not implementation failure. motivates supervised approaches.

**gan:** dcgan produces visually realistic chest x-rays by epoch 5, stable equilibrium throughout. best checkpoint epoch 10 (fid 89.56).

**diffusion:** ddpm with ddim (50 steps) outperforms dcgan in fid (64.09 vs 89.56). trained at 128×128 due to vram.

**u-net:** 0.40 dice on opacity images (honest metric), 0.87 on normal. overall 0.76 inflated by all-black masks — reported separately.

**clip:** general clip (auc 0.51) vs biomedclip (auc 0.84) zero-shot. +0.33 auc gap isolates domain pretraining. accuracy/auc trap documented.

---

## code style
- lowercase comments, no decorative dividers, no artificial spacing
- comments only where genuinely needed
- code should look human-written

---

## session behavior note
when context window reaches ~90% full, generate updated handoff before continuing.