import os
import sys
import json
import numpy as np

sys.path.insert(0, os.getcwd())

import torch
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score)

from configs.config import (IMAGE_SIZE, BATCH_SIZE, DEVICE, SEED)
from src.utils.dataset import get_dataloaders
from src.models.classifier import build_classifier


def eval_classifier():
    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # load val split
    _, val_loader, _ = get_dataloaders()
    print(f"val samples: {len(val_loader.dataset)}")

    # load checkpoint
    ckpt_path = 'outputs/checkpoints/best_resnet18.pth'
    model = build_classifier(backbone='resnet18', num_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
    model.eval()
    print(f"loaded: {ckpt_path}")

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label']
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of pneumonia class
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['normal', 'pneumonia'], output_dict=True)
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # confusion matrix: cm[true][pred]
    tn, fp, fn, tp = cm.ravel()

    print(f"\nval accuracy : {acc*100:.2f}%")
    print(f"auc-roc : {auc:.4f}")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['normal', 'pneumonia'])}")

    # build output dict
    results = {
        "final_val_accuracy": round(acc * 100, 4),
        "auc_roc": round(float(auc), 4),
        "best_epoch": 11,
        "stopped_epoch": 16,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "labels": ["normal", "pneumonia"]
        },
        "per_class": {
            "normal": {
                "precision": round(report["normal"]["precision"], 4),
                "recall": round(report["normal"]["recall"], 4),
                "f1": round(report["normal"]["f1-score"], 4),
                "support": int(report["normal"]["support"])
            },
            "pneumonia": {
                "precision": round(report["pneumonia"]["precision"], 4),
                "recall": round(report["pneumonia"]["recall"], 4),
                "f1": round(report["pneumonia"]["f1-score"], 4),
                "support": int(report["pneumonia"]["support"])
            }
        },
        "roc_curve": {
            # downsample to 100 points so json isn't huge
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//100)]],
            "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//100)]]
        },
        "history": None
    }

    out_path = 'dashboard/public/results/classifier.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nsaved: {out_path}")


if __name__ == '__main__':
    eval_classifier()