#ddpm for chest x-ray synthesis. two pieces here:
#UNet - a time-conditioned denoiser that predicts the noise added to an image
#Diffusion - holds the noise schedule and the forward (q_sample) / reverse (ddim) processes
#training predicts noise at a random timestep (epsilon objective). sampling uses ddim so we
#can generate in ~50 steps instead of the 1000 used during training - this is what makes
#cpu inference on the dashboard feasible.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    #encodes the scalar timestep t into a vector the network can condition on,
    #same idea as positional encodings in transformers.
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    #conv block with the timestep embedding injected between the two convs.
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    #3-level encoder/decoder denoiser. no self-attention - keeps memory low enough for 4gb.
    def __init__(self, base=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_conv = nn.Conv2d(1, base, 3, padding=1)

        self.down1 = ResBlock(base, base, time_dim)
        self.ds1 = nn.Conv2d(base, base, 4, 2, 1)
        self.down2 = ResBlock(base, base * 2, time_dim)
        self.ds2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)
        self.down3 = ResBlock(base * 2, base * 4, time_dim)
        self.ds3 = nn.Conv2d(base * 4, base * 4, 4, 2, 1)

        self.mid = ResBlock(base * 4, base * 4, time_dim)

        self.us3 = nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1)
        self.up3 = ResBlock(base * 8, base * 2, time_dim)
        self.us2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1)
        self.up2 = ResBlock(base * 4, base, time_dim)
        self.us1 = nn.ConvTranspose2d(base, base, 4, 2, 1)
        self.up1 = ResBlock(base * 2, base, time_dim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)
        s1 = self.down1(x, t)
        x = self.ds1(s1)
        s2 = self.down2(x, t)
        x = self.ds2(s2)
        s3 = self.down3(x, t)
        x = self.ds3(s3)
        x = self.mid(x, t)
        x = self.us3(x)
        x = self.up3(torch.cat([x, s3], dim=1), t)
        x = self.us2(x)
        x = self.up2(torch.cat([x, s2], dim=1), t)
        x = self.us1(x)
        x = self.up1(torch.cat([x, s1], dim=1), t)
        return self.out_conv(F.silu(self.out_norm(x)))


class Diffusion:
    def __init__(self, timesteps, img_size, device):
        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device

        #linear beta schedule - standard ddpm. betas control how much noise is added per step.
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.acp = acp.to(device)
        self.sqrt_acp = torch.sqrt(acp).to(device)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp).to(device)

    def q_sample(self, x0, t, noise):
        #forward process: jump straight to a noised version of x0 at timestep t in closed form.
        a = self.sqrt_acp[t][:, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None]
        return a * x0 + b * noise

    def p_losses(self, model, x0):
        #pick a random timestep per image, noise it, ask the model to predict the noise back.
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def ddim_sample(self, model, n, ddim_steps=50):
        #deterministic ddim (eta=0). walks a subset of timesteps from noise back to an image.
        model.eval()
        steps = torch.linspace(self.timesteps - 1, 0, ddim_steps).long().to(self.device)
        x = torch.randn(n, 1, self.img_size, self.img_size, device=self.device)

        for i in range(len(steps)):
            t = steps[i]
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            acp_t = self.acp[t]
            x0_pred = (x - torch.sqrt(1 - acp_t) * pred_noise) / torch.sqrt(acp_t)
            x0_pred = x0_pred.clamp(-1, 1)

            if i < len(steps) - 1:
                acp_next = self.acp[steps[i + 1]]
                x = torch.sqrt(acp_next) * x0_pred + torch.sqrt(1 - acp_next) * pred_noise
            else:
                x = x0_pred
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(base=64).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 1, 128, 128).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    out = model(x, t)
    print(f"in: {x.shape} -> out: {out.shape}")