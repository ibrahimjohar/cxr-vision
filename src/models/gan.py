"""
GAN implemented:
    - DCGAN Generator (noise → X-ray)
    - DCGAN Discriminator (X-ray → real/fake)
    - GAN training step with alternating updates
    - Non-saturating generator loss
architecture follows DCGAN guidelines:
    - Strided conv instead of pooling
    - BatchNorm in both G and D
    - ReLU in G, LeakyReLU in D
    - No fully connected hidden layers
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from configs.config import GAN_LATENT_DIM

#maps noise z (B, latent_dim) -> fake x-ray (B,1,224,224)
#uses transposed convolutions to upsample from small spatial dims
class Generator(nn.Module):
    def __init__(self, latent_dim=GAN_LATENT_DIM):
        super().__init__()
        #input: (B, latent_dim) → reshape to (B, 512, 7, 7)
        self.net = nn.Sequential(nn.Linear(latent_dim, 512 * 7 * 7))
        self.conv = nn.Sequential(
            # (B, 512, 7, 7) → (B, 256, 14, 14)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # → (B, 128, 28, 28)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # → (B, 64, 56, 56)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # → (B, 32, 112, 112)
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # → (B, 1, 224, 224)
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()   # output in [-1, 1]
        )

    def forward(self, z):
        x = self.net(z)
        x = x.view(x.size(0), 512, 7, 7)
        return self.conv(x)

#maps x-ray (B,1,224,224) -> real/fake probability (B,1)
#uses strided convolutions to downsample and leakyReLU to allow small gradients for -ve activations
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 1, 224, 224) → (B, 32, 112, 112)
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → (B, 64, 56, 56)
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # → (B, 128, 28, 28)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # → (B, 256, 14, 14)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # → (B, 1, 7, 7) → flatten → sigmoid
            nn.Conv2d(256, 1, 4, 2, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),    # global average pool → (B, 1, 1, 1)
            nn.Flatten(),               # → (B, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

#initialize weights from N(0,0.02) as per DCGAN paper
#applied to conv and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def build_gan(latent_dim=GAN_LATENT_DIM, device="cpu"):
    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    total_g = sum(p.numel() for p in G.parameters())
    total_d = sum(p.numel() for p in D.parameters())
    print(f"[GAN] Generator params: {total_g:,} | Discriminator params: {total_d:,}")
    return G, D

#non-saturating generator loss: maximize log(D(G(z))) = minimize -log(D(G(z)))
#better gradients early in training vs original log(1-D(G(z)))
def generator_loss(D_fake):
    return -torch.log(D_fake + 1e-8).mean()

#discriminator loss: maximize log(D(x)) + log(1-D(G(z))) = minimize -[log(D(x)) + log(1-D(G(z)))]
def discriminator_loss(D_real, D_fake):
    real_loss = -torch.log(D_real + 1e-8).mean()
    fake_loss = -torch.log(1 - D_fake + 1e-8).mean()
    return real_loss + fake_loss


if __name__ == "__main__":
    G, D = build_gan()
    z = torch.randn(4, GAN_LATENT_DIM)
    fake = G(z)
    score = D(fake)
    print(f"Noise shape: {z.shape}")
    print(f"Generated shape: {fake.shape}")
    print(f"D(G(z)) scores: {score.squeeze()}")
