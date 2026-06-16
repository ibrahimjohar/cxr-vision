#zero-shot evaluation comparing openai clip vs microsoft biomedclip.
#neither model is trained on rsna - both are evaluated zero-shot.
#openai clip: trained on 400M natural image-text pairs, no medical data.
#biomedclip: trained on 15M biomedical figure-caption pairs from pubmed central.
#the gap between the two directly shows how much domain-specific pretraining matters.

import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import clip
from open_clip import create_model_from_pretrained, get_tokenizer

from configs.config import FIGURES_DIR, DEVICE
from src.utils.dataset import build_rsna_dataset

#openai clip prompts - plain phrasing worked best in initial eval
CLIP_PROMPTS = [
    "a normal chest x-ray",
    "a chest x-ray with pneumonia",
]

#biomedclip uses 'this is a photo of ' prefix - that's the template from its training
BIOMED_PROMPTS = [
    "this is a photo of a normal chest x-ray with clear lungs",
    "this is a photo of a chest x-ray showing pulmonary opacity and pneumonia",
]


def load_test_data():
    _, _, test_p, _, _, test_l = build_rsna_dataset()
    return test_p, np.array(test_l)


@torch.no_grad()
def eval_openai_clip(device, paths, labels, batch_size=64):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("[openai clip] loaded vit-b/32")

    text_tokens = clip.tokenize(CLIP_PROMPTS).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    pneumonia_probs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  openai clip", leave=False):
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]).to(device)
        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * feats @ text_features.t()
        probs = logits.softmax(dim=-1)
        pneumonia_probs.extend(probs[:, 1].cpu().numpy())

    pneumonia_probs = np.array(pneumonia_probs)
    auc = roc_auc_score(labels, pneumonia_probs)
    acc = accuracy_score(labels, (pneumonia_probs > 0.5).astype(int))
    print(f"  auc={auc:.4f} | acc={acc:.4f}")
    return auc, acc


@torch.no_grad()
def eval_biomedclip(device, paths, labels, batch_size=64):
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    print("[biomedclip] loaded pubmedbert-256 vit-b/16")

    text_tokens = tokenizer(BIOMED_PROMPTS).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    pneumonia_probs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  biomedclip", leave=False):
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]).to(device)
        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = (feats @ text_features.t()) * model.logit_scale.exp()
        probs = logits.softmax(dim=-1)
        pneumonia_probs.extend(probs[:, 1].cpu().numpy())

    pneumonia_probs = np.array(pneumonia_probs)
    auc = roc_auc_score(labels, pneumonia_probs)
    acc = accuracy_score(labels, (pneumonia_probs > 0.5).astype(int))
    print(f"  auc={auc:.4f} | acc={acc:.4f}")
    return auc, acc


def plot_comparison(clip_auc, clip_acc, biomed_auc, biomed_acc, cnn_auc=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    models = ["openai clip\n(general)", "biomedclip\n(medical)"]
    aucs = [clip_auc, biomed_auc]
    accs = [clip_acc, biomed_acc]

    if cnn_auc is not None:
        models.append("resnet18\n(supervised)")
        aucs.append(cnn_auc)
        accs.append(None)                               #accuracy not plotted for cnn here

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x[:2] - w/2, aucs[:2], w, label="auc", color="steelblue")
    ax.bar(x[:2] + w/2, accs, w, label="accuracy", color="tomato")
    if cnn_auc is not None:
        ax.bar(x[2], cnn_auc, w*2, label="cnn auc (reference)", color="seagreen", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("zero-shot clip vs domain-specific biomedclip vs supervised cnn")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "clip_comparison.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved comparison -> {path}")


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    paths, labels = load_test_data()
    print(f"[data] {len(paths)} test images\n")

    print("[evaluating openai clip]")
    clip_auc, clip_acc = eval_openai_clip(device, paths, labels)

    # free vram between models
    torch.cuda.empty_cache()

    print("\n[evaluating biomedclip]")
    biomed_auc, biomed_acc = eval_biomedclip(device, paths, labels)

    print(f"\n[summary]")
    print(f"openai clip — auc={clip_auc:.4f} acc={clip_acc:.4f}  (general, no medical training)")
    print(f"biomedclip — auc={biomed_auc:.4f} acc={biomed_acc:.4f}  (15M biomedical image-text pairs)")
    print(f"delta auc: {biomed_auc - clip_auc:+.4f}")

    # include supervised cnn reference line (85% acc ≈ reported val acc, rough auc reference)
    plot_comparison(clip_auc, clip_acc, biomed_auc, biomed_acc, cnn_auc=None)
    return clip_auc, biomed_auc


if __name__ == "__main__":
    main()