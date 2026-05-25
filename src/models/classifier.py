"""
implemented:
    - Pretrained ResNet backbone (transfer learning)
    - Custom classification head with dropout
    - Model factory supporting multiple backbones
this is the CNN classifier stage of the pipeline.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torchvision.models as models
from configs.config import NUM_CLASSES, DROPOUT_RATE, CNN_BACKBONE, PRETRAINED


#Classifier
class XrayClassifier(nn.Module):
    #Transfer learning classifier for chest X-ray pathology detection.
    #uses a pretrained CNN backbone with a custom classification head.
    #the backbone (ResNet/DenseNet) acts as a feature extractor trained on ImageNet - it already knows edges, textures, shapes.
    #we fine-tune it on X-ray data so those features adapt to medical imaging.
    #the classification head maps extracted features → class probabilities.
    def __init__(self, backbone=CNN_BACKBONE, num_classes=NUM_CLASSES, pretrained=PRETRAINED, dropout=DROPOUT_RATE):
        super(XrayClassifier, self).__init__()

        self.backbone_name = backbone
        self.backbone, in_features = self._build_backbone(backbone, pretrained)

        #classification head: Dropout → Linear → (softmax applied in loss)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

    #load backbone and remove its final classification layer.
    #returns (backbone_without_head, num_output_features).
    def _build_backbone(self, backbone, pretrained):
        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "resnet18":
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()            #remove original FC layer

        elif backbone == "resnet50":
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()

        elif backbone == "densenet121":
            model = models.densenet121(weights=weights)
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()

        else:
            raise ValueError(f"unsupported backbone: {backbone}. "
                             f"choose from: resnet18, resnet50, densenet121")

        return model, in_features

    #forward pass: x (B,3,H,W) -> backbone features (B, in_features) -> classifier head (B, num_classes)
    def forward(self, x):
        #extract features
        features = self.backbone(x)     
        #classify
        logits = self.classifier(features)
        return logits

    #extract feature embeddings without classification - used for CLIP-style evaluation & visualization
    def get_feature_embeddings(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return features


#model info
def count_parameters(model):
    #count trainable parameters in a model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

#factory function - returns a ready-to-train classifier
def build_classifier(backbone=CNN_BACKBONE, num_classes=NUM_CLASSES):
    model = XrayClassifier(backbone=backbone, num_classes=num_classes)
    total, trainable = count_parameters(model)
    print(f"[Model] {backbone} | Total params: {total:,} | Trainable: {trainable:,}")
    return model


if __name__ == "__main__":
    model = build_classifier("resnet18")
    dummy_input = torch.randn(4, 3, 224, 224)   #batch of 4 images
    output = model(dummy_input)
    print(f"input shape: {dummy_input.shape}")
    print(f"output shape: {output.shape}")     #should be (4, 2)
    print(f"output (logits): {output}")
