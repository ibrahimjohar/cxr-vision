"""
src/models/attention_unet.py

Weeks 10-11 of the course implemented:
    - U-Net encoder-decoder architecture
    - Attention gates on skip connections
    - Each attention gate learns to highlight relevant spatial regions

The attention gate computes a gating coefficient for each spatial location.
High coefficient = this region is relevant for the current decoder state.
Low coefficient = suppress this region (likely background).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import UNET_FEATURES


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and ReLU. Standard U-Net block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention gate for U-Net skip connections.

    Takes:
        g — gating signal from decoder (coarser, semantic)
        x — skip connection from encoder (finer, spatial)
    Returns:
        x weighted by learned attention coefficients

    The gate learns which spatial regions in the encoder feature map
    are relevant given the decoder's current state.
    This replaces blind concatenation of encoder features with
    selectively weighted, context-aware features.
    """

    def __init__(self, F_g, F_l, F_int):
        """
        F_g  : channels in gating signal (from decoder)
        F_l  : channels in skip connection (from encoder)
        F_int: intermediate channels in attention computation
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()                # attention coefficient in [0,1]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Upsample g to match x spatial size
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)       # add gating signal and skip connection
        psi = self.psi(psi)            # compute spatial attention map
        return x * psi                 # weight skip connection by attention map


# ─── Attention U-Net ──────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net for chest X-ray segmentation / feature extraction.

    Encoder path: progressively downsample and extract features
    Decoder path: progressively upsample, using attention-gated skip connections
    Output: segmentation map (B, num_classes, H, W)
    """

    def __init__(self, in_channels=1, num_classes=1,
                 features=UNET_FEATURES):
        super().__init__()

        self.encoder1 = ConvBlock(in_channels,  features[0])
        self.encoder2 = ConvBlock(features[0],  features[1])
        self.encoder3 = ConvBlock(features[1],  features[2])
        self.encoder4 = ConvBlock(features[2],  features[3])

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        # Attention gates — F_g must match upsampled decoder channels
        self.attn4 = AttentionGate(features[3],   features[3], features[3]//2)
        self.attn3 = AttentionGate(features[2],   features[2], features[2]//2)
        self.attn2 = AttentionGate(features[1],   features[1], features[1]//2)
        self.attn1 = AttentionGate(features[0],   features[0], features[0]//2)

        # Decoder
        self.up4     = nn.ConvTranspose2d(features[3]*2, features[3], 2, 2)
        self.decoder4= ConvBlock(features[3]*2, features[3])

        self.up3     = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.decoder3= ConvBlock(features[2]*2, features[2])

        self.up2     = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.decoder2= ConvBlock(features[1]*2, features[1])

        self.up1     = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.decoder1= ConvBlock(features[0]*2, features[0])

        # Final output
        self.output  = nn.Conv2d(features[0], num_classes, 1)
        if num_classes == 1:
            self.act = nn.Sigmoid()     # binary segmentation
        else:
            self.act = nn.Softmax(dim=1)

    def forward(self, x):
        # ── Encoder ───────────────────────────────────────────────────────────
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # ── Bottleneck ────────────────────────────────────────────────────────
        b  = self.bottleneck(self.pool(e4))

        # ── Decoder with Attention Gates ──────────────────────────────────────
        d4 = self.up4(b)
        e4 = self.attn4(g=d4, x=e4)        # attention-weight encoder skip
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.attn3(g=d3, x=e3)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.attn2(g=d2, x=e2)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.attn1(g=d1, x=e1)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        return self.act(self.output(d1))


if __name__ == "__main__":
    model = AttentionUNet(in_channels=1, num_classes=1)
    total = sum(p.numel() for p in model.parameters())
    print(f"[AttentionUNet] Total parameters: {total:,}")

    x   = torch.randn(2, 1, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # should be (2, 1, 224, 224)
