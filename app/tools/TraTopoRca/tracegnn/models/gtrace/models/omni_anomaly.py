import torch
import torch.nn as nn
import math
from typing import Dict

class HostVAE(nn.Module):
    """
    基于主机状态快照（非序列）的简单 VAE
    """
    def __init__(self, in_dim: int, hidden: int, latent: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_hat = self.dec(z)
        recon = torch.mean((x - x_hat) ** 2, dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return {"recon": recon, "kl": kl}


class HostOmniAnomaly(nn.Module):
    """
    基于 GRU + VAE 的时序异常检测模型 (OmniAnomaly Simplified)
    输入 X: [B, T, D]
    """
    def __init__(self, d_in: int, h: int, z: int):
        super().__init__()
        self.enc = nn.GRU(input_size=d_in, hidden_size=h, batch_first=True)
        self.mu_z = nn.Linear(h, z)
        self.logvar_z = nn.Linear(h, z)
        self.dec = nn.GRU(input_size=z, hidden_size=h, batch_first=True)
        self.mu_x = nn.Linear(h, d_in)
        self.logvar_x = nn.Linear(h, d_in)

    def forward(self, X: torch.Tensor, beta_kl: float, mask=None):
        # X: [B, T, D]
        B, T, D = X.shape
        h, _ = self.enc(X)  # [B,T,H]
        mu_z, logvar_z = self.mu_z(h), self.logvar_z(h)
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        Z = mu_z + std_z * eps  # [B,T,Z]
        h_dec, _ = self.dec(Z)  # [B,T,H]
        mu_x = self.mu_x(h_dec)  # [B,T,D]
        logvar_x = self.logvar_x(h_dec)  # [B,T,D]

        nll = 0.5 * (math.log(2 * math.pi) + logvar_x + (X - mu_x) ** 2 / torch.exp(logvar_x))  # [B,T,D]
        nll = nll.sum(dim=2)  # [B,T]
        kl_t = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=2)  # [B,T]

        if mask is not None:
            nll = nll * mask
            kl_t = kl_t * mask
            denom = mask.sum(dim=1).clamp(min=1.0)  # [B]
        else:
            denom = torch.tensor(float(T), device=X.device).repeat(B)

        nll_host = (nll.sum(dim=1) + beta_kl * kl_t.sum(dim=1)) / denom  # [B]
        return nll_host.mean(), nll_host