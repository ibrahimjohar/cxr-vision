"""
fastapi inference server for cxr-vision.

endpoints:
    POST /predict: classifier, attention u-net, and vae on an uploaded x-ray
    POST /clip: openai clip and biomedclip zero-shot scores on an uploaded x-ray
    POST /generate/gan: one dcgan sample
    POST /generate/diffusion: one ddim diffusion sample

run from the project root:
    uvicorn api.main:app --reload
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import io
import base64
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import torchvision.transforms as transforms
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.models.classifier import XrayClassifier
from src.models.vae import VAE
from src.models.attention_unet import AttentionUNet
from src.models.gan import Generator
from src.models.diffusion import UNet as DiffusionUNet, Diffusion
from configs.config import (
    CNN_BACKBONE, NUM_CLASSES, DROPOUT_RATE,
    VAE_LATENT_DIM, UNET_FEATURES, IMAGE_SIZE, CHECKPOINTS_DIR,
    GAN_LATENT_DIM, DIFFUSION_BASE_CH, DIFFUSION_TIMESTEPS,
    DIFFUSION_IMG_SIZE, DIFFUSION_DDIM_STEPS,
)

DEVICE = "cpu" # modal free tier is cpu only

app = FastAPI(title="cxr-vision inference api")

#rate limiting keyed by client ip. per-container in memory, not shared across
#containers, but sufficient for this project's traffic level.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # tighten to the vercel domain once deployed
    allow_methods=["POST"],
    allow_headers=["*"],
)

#classifier: rgb, imagenet normalized. matches src/utils/dataset.py get_transforms("val") exactly.
classifier_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#vae and u-net: grayscale, [0,1] range, matches the sigmoid decoder/mask outputs.
#assumption: train_vae.py and train_unet.py weren't available to confirm this is
#exactly the training time transform. verify before trusting these outputs.
grayscale_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

#zero-shot prompts, exactly as validated in src/evaluation/clip_eval.py
CLIP_PROMPTS = [
    "a normal chest x-ray",
    "a chest x-ray with pneumonia",
]
BIOMED_PROMPTS = [
    "this is a photo of a normal chest x-ray with clear lungs",
    "this is a photo of a chest x-ray showing pulmonary opacity and pneumonia",
]


def load_checkpoint(model, filename, state_key="model_state"):
    path = os.path.join(CHECKPOINTS_DIR, filename)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model


#load every model once at startup, not per request.
#pretrained=False on the classifier backbone on purpose, we're about to overwrite
#every weight with our fine tuned checkpoint so there's no reason to pay for an
#imagenet download on every cold start.
classifier = load_checkpoint(
    XrayClassifier(backbone=CNN_BACKBONE, num_classes=NUM_CLASSES, pretrained=False, dropout=DROPOUT_RATE),
    "best_resnet18.pth",
)
vae = load_checkpoint(VAE(latent_dim=VAE_LATENT_DIM), "best_vae.pth")
unet = load_checkpoint(AttentionUNet(in_channels=1, num_classes=1, features=UNET_FEATURES), "best_unet.pth")

#gan checkpoint stores G and D under separate keys, generation only needs G
gan_generator = load_checkpoint(Generator(latent_dim=GAN_LATENT_DIM), "gan_epoch010.pth", state_key="G_state")

#diffusion checkpoint uses the same model_state convention as classifier/vae/unet
diffusion_model = load_checkpoint(DiffusionUNet(base=DIFFUSION_BASE_CH), "diffusion_epoch010.pth")
diffusion_process = Diffusion(DIFFUSION_TIMESTEPS, DIFFUSION_IMG_SIZE, DEVICE)

#clip and biomedclip, text features precomputed once since the prompts are fixed
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
with torch.no_grad():
    clip_text_features = clip_model.encode_text(clip.tokenize(CLIP_PROMPTS).to(DEVICE))
    clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)

biomed_model, biomed_preprocess = create_model_from_pretrained(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
biomed_model = biomed_model.to(DEVICE)
biomed_model.eval()
biomed_tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
with torch.no_grad():
    biomed_text_features = biomed_model.encode_text(biomed_tokenizer(BIOMED_PROMPTS).to(DEVICE))
    biomed_text_features = biomed_text_features / biomed_text_features.norm(dim=-1, keepdim=True)


def image_to_base64_png(arr: np.ndarray) -> str:
    #uint8 array (HxW or HxWx3) -> base64 png string the frontend can render directly
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def read_upload_as_rgb(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not read uploaded file as an image")


@app.get("/")
async def health():
    return {"status": "ok", "models_loaded": ["classifier", "vae", "unet", "gan", "diffusion", "clip", "biomedclip"]}


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    pil_img = read_upload_as_rgb(raw)

    with torch.no_grad():
        #classifier probabilities: index 0 is normal, index 1 is pneumonia
        cls_input = classifier_transform(pil_img).unsqueeze(0).to(DEVICE)
        logits = classifier(cls_input)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

        #u-net and vae share the same grayscale preprocessing
        gray_input = grayscale_transform(pil_img).unsqueeze(0).to(DEVICE)

        mask = unet(gray_input).squeeze().cpu().numpy() # (224,224) in [0,1]
        mask_img = (mask * 255).astype(np.uint8)

        recon, mu, logvar = vae(gray_input)
        recon_np = recon.squeeze().cpu().numpy()
        recon_img = (recon_np * 255).astype(np.uint8)
        recon_error = torch.nn.functional.mse_loss(recon, gray_input).item()

    return {
        "classifier": {
            "normal": round(probs[0], 4),
            "pneumonia": round(probs[1], 4),
        },
        "unet": {
            "mask_png_base64": image_to_base64_png(mask_img),
        },
        "vae": {
            "reconstruction_png_base64": image_to_base64_png(recon_img),
            "reconstruction_mse": round(recon_error, 6),
        },
    }


@app.post("/clip")
@limiter.limit("10/minute")
async def clip_predict(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    pil_img = read_upload_as_rgb(raw)

    with torch.no_grad():
        clip_input = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
        clip_feats = clip_model.encode_image(clip_input)
        clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True)
        clip_logits = clip_model.logit_scale.exp() * clip_feats @ clip_text_features.t()
        clip_probs = clip_logits.softmax(dim=-1).squeeze(0).tolist()

        biomed_input = biomed_preprocess(pil_img).unsqueeze(0).to(DEVICE)
        biomed_feats = biomed_model.encode_image(biomed_input)
        biomed_feats = biomed_feats / biomed_feats.norm(dim=-1, keepdim=True)
        biomed_logits = (biomed_feats @ biomed_text_features.t()) * biomed_model.logit_scale.exp()
        biomed_probs = biomed_logits.softmax(dim=-1).squeeze(0).tolist()

    return {
        "openai_clip": {
            "normal": round(clip_probs[0], 4),
            "pneumonia": round(clip_probs[1], 4),
        },
        "biomedclip": {
            "normal": round(biomed_probs[0], 4),
            "pneumonia": round(biomed_probs[1], 4),
        },
    }


@app.post("/generate/gan")
@limiter.limit("15/minute")
async def generate_gan(request: Request):
    with torch.no_grad():
        z = torch.randn(1, GAN_LATENT_DIM).to(DEVICE)
        fake = gan_generator(z).squeeze().cpu().numpy()
        img = (((fake + 1) / 2) * 255).astype(np.uint8) # [-1,1] -> [0,255]

    return {"image_png_base64": image_to_base64_png(img)}


@app.post("/generate/diffusion")
@limiter.limit("5/minute")
async def generate_diffusion(request: Request):
    with torch.no_grad():
        sample = diffusion_process.ddim_sample(diffusion_model, 1, ddim_steps=DIFFUSION_DDIM_STEPS)
        sample = sample.squeeze().cpu().numpy()
        img = (((sample + 1) / 2) * 255).astype(np.uint8) # [-1,1] -> [0,255]

    return {"image_png_base64": image_to_base64_png(img)}