"""
src/models/vae.py

Week 7 of the course implemented:
    - Convolutional encoder → (mu, logvar)
    - Reparameterization trick
    - Convolutional decoder
    - VAE loss = Reconstruction Loss + KL Divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import VAE_LATENT_DIM, IMAGE_SIZE


class ConvEncoder(nn.Module):
    """
    Convolutional encoder.
    Maps (B, 1, H, W) → (B, latent_dim) as mu and logvar.
    Takes grayscale X-rays (1 channel).
    """

    def __init__(self, latent_dim=VAE_LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   nn.ReLU(),   # 224→112
            nn.Conv2d(32, 64, 4, 2, 1),  nn.ReLU(),   # 112→56
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),   # 56→28
            nn.Conv2d(128, 256, 4, 2, 1),nn.ReLU(),   # 28→14
        )
        self.flatten_size = 256 * 14 * 14
        self.fc_mu     = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x   = self.encoder(x)
        x   = x.view(x.size(0), -1)
        mu  = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ConvDecoder(nn.Module):
    """
    Convolutional decoder.
    Maps (B, latent_dim) → (B, 1, H, W).
    """

    def __init__(self, latent_dim=VAE_LATENT_DIM):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, 256 * 14 * 14)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),   # 14→28
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(),   # 28→56
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(),   # 56→112
            nn.ConvTranspose2d(32,  1,   4, 2, 1), nn.Sigmoid() # 112→224
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 14, 14)
        return self.decoder(x)


class VAE(nn.Module):
    """
    Full Variational Autoencoder.

    Forward returns (reconstruction, mu, logvar).
    Use vae_loss() to compute training loss.
    For generation: sample z ~ N(0,1) and call decoder directly.
    """

    def __init__(self, latent_dim=VAE_LATENT_DIM):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
            z = mu + sigma * epsilon
            epsilon ~ N(0, 1)
        Makes sampling differentiable for backprop.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)   # sigma = exp(0.5 * log(sigma^2))
            eps = torch.randn_like(std)      # epsilon ~ N(0,1)
            return mu + eps * std
        else:
            return mu                        # at inference, use mean directly

    def forward(self, x):
        mu, logvar        = self.encoder(x)
        z                 = self.reparameterize(mu, logvar)
        reconstruction    = self.decoder(z)
        return reconstruction, mu, logvar

    def generate(self, n_samples=16, device="cpu"):
        """Sample from prior N(0,1) and decode → new images."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            return self.decoder(z)

    def reconstruct(self, x):
        """Encode then decode a real image."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z          = self.reparameterize(mu, logvar)
            return self.decoder(z), mu


def vae_loss(reconstruction, original, mu, logvar, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + beta × KL Divergence

    Reconstruction: MSE between input and output pixels
    KL:             -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    beta:           weight on KL term (beta-VAE, default 1 = standard VAE)
    """
    recon_loss = F.mse_loss(reconstruction, original, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


if __name__ == "__main__":
    vae   = VAE(latent_dim=128)
    x     = torch.randn(4, 1, 224, 224)
    recon, mu, logvar = vae(x)
    loss, rl, kl = vae_loss(recon, x, mu, logvar)
    print(f"Input:          {x.shape}")
    print(f"Reconstruction: {recon.shape}")
    print(f"mu:             {mu.shape}")
    print(f"Total loss: {loss.item():.4f} | Recon: {rl.item():.4f} | KL: {kl.item():.4f}")
