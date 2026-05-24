"""
src/models/classifier.py

Week 5-6 of the course implemented:
    - Pretrained ResNet backbone (transfer learning)
    - Custom classification head with dropout
    - Model factory supporting multiple backbones

This is the CNN classifier stage of the pipeline.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torchvision.models as models
from configs.config import NUM_CLASSES, DROPOUT_RATE, CNN_BACKBONE, PRETRAINED


# ─── Classifier ───────────────────────────────────────────────────────────────

class XrayClassifier(nn.Module):
    """
    Transfer learning classifier for chest X-ray pathology detection.
    Uses a pretrained CNN backbone with a custom classification head.

    The backbone (ResNet/DenseNet) acts as a feature extractor trained on
    ImageNet — it already knows edges, textures, shapes. We fine-tune it
    on X-ray data so those features adapt to medical imaging.

    The classification head maps extracted features → class probabilities.
    """

    def __init__(self, backbone=CNN_BACKBONE, num_classes=NUM_CLASSES,
                 pretrained=PRETRAINED, dropout=DROPOUT_RATE):
        super(XrayClassifier, self).__init__()

        self.backbone_name = backbone
        self.backbone, in_features = self._build_backbone(backbone, pretrained)

        # Classification head: Dropout → Linear → (softmax applied in loss)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

    def _build_backbone(self, backbone, pretrained):
        """
        Load backbone and remove its final classification layer.
        Returns (backbone_without_head, num_output_features).
        """
        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "resnet18":
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()            # remove original FC layer

        elif backbone == "resnet50":
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()

        elif backbone == "densenet121":
            model = models.densenet121(weights=weights)
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                             f"Choose from: resnet18, resnet50, densenet121")

        return model, in_features

    def forward(self, x):
        """
        Forward pass:
            x (B, 3, H, W) → backbone features (B, in_features)
                           → classifier head (B, num_classes)
        """
        features = self.backbone(x)     # extract features
        logits   = self.classifier(features)   # classify
        return logits

    def get_feature_embeddings(self, x):
        """
        Extract feature embeddings without classification.
        Used for CLIP-style evaluation and visualization.
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features


# ─── Model Info ───────────────────────────────────────────────────────────────

def count_parameters(model):
    """Count trainable parameters in a model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_classifier(backbone=CNN_BACKBONE, num_classes=NUM_CLASSES):
    """
    Factory function — returns a ready-to-train classifier.
    Usage:
        from src.models.classifier import build_classifier
        model = build_classifier()
    """
    model = XrayClassifier(backbone=backbone, num_classes=num_classes)
    total, trainable = count_parameters(model)
    print(f"[Model] {backbone} | Total params: {total:,} | Trainable: {trainable:,}")
    return model


if __name__ == "__main__":
    model = build_classifier("resnet18")
    dummy_input = torch.randn(4, 3, 224, 224)   # batch of 4 images
    output = model(dummy_input)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")     # should be (4, 2)
    print(f"Output (logits): {output}")
