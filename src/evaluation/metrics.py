import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from configs.config import FIGURES_DIR


def compute_classification_metrics(all_labels, all_probs, threshold=0.5):
    #all_labels: list/array of true binary labels
    #all_probs: list/array of predicted probabilities for the positive class
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= threshold).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nclassification metrics:")
    print(f"auc-roc: {auc:.4f}")
    print(f"f1 score: {f1:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['normal', 'pneumonia'])}")

    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall, "cm": cm}


def compute_dice_score(pred_mask, true_mask, smooth=1.0):
    #dice score = 2 * intersection / (pred + target)
    #standard metric for segmentation quality - ranges from 0 (no overlap) to 1 (perfect)
    pred = pred_mask.view(-1).float()
    true = true_mask.view(-1).float()
    intersection = (pred * true).sum()
    return ((2 * intersection + smooth) / (pred.sum() + true.sum() + smooth)).item()


def compute_vae_anomaly_scores(model, loader, device):
    #reconstruction error per image - higher error = more anomalous
    #used to evaluate vae anomaly detection without needing labels during training
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            recon, _, _ = model(images)
            #mse per image (mean over pixels)
            error = ((recon - images) ** 2).mean(dim=[1, 2, 3])
            scores.extend(error.cpu().numpy())
            labels.extend(batch["label"].numpy())
    return np.array(scores), np.array(labels)


def plot_confusion_matrix(cm, save=True):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["normal", "pneumonia"],
                yticklabels=["normal", "pneumonia"], ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("confusion matrix")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"saved confusion matrix -> {path}")
    else:
        plt.show()


def plot_roc_curve(all_labels, all_probs, save=True):
    from sklearn.metrics import roc_curve
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"auc = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("roc curve")
    ax.legend()
    plt.tight_layout()
    
    if save:
        path = os.path.join(FIGURES_DIR, "roc_curve.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"saved roc curve -> {path}")
    else:
        plt.show()


def plot_anomaly_scores(scores, labels, save=True):
    #overlay reconstruction error distributions for normal vs pneumonia
    #a good vae should show separation between the two distributions
    os.makedirs(FIGURES_DIR, exist_ok=True)
    normal_scores = scores[labels == 0]
    pneumonia_scores = scores[labels == 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_scores, bins=30, alpha=0.6, label="normal", color="steelblue")
    ax.hist(pneumonia_scores, bins=30, alpha=0.6, label="pneumonia", color="tomato")
    ax.set_xlabel("reconstruction error")
    ax.set_ylabel("count")
    ax.set_title("vae anomaly score distribution")
    ax.legend()
    plt.tight_layout()
    
    if save:
        path = os.path.join(FIGURES_DIR, "vae_anomaly_scores.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"saved anomaly score plot -> {path}")
    else:
        plt.show()


def plot_training_curves(history, model_name, save=True):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{model_name} loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="train")
        axes[1].plot(history["val_acc"], label="val")
        axes[1].set_title(f"{model_name} accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].legend()
    plt.tight_layout()
    
    if save:
        path = os.path.join(FIGURES_DIR, f"{model_name}_training_curves.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"saved training curves -> {path}")
    else:
        plt.show()