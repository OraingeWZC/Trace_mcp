import torch
import torch.nn as nn
import math

class AnomalyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.W_o = nn.Linear(d_model * n_heads, d_model, bias=False)
        self.sigma = nn.Parameter(torch.ones(n_heads, 1, 1))

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_model).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_model).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_model).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.d_model)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        series_attn = torch.softmax(scores, dim=-1)
        
        indices = torch.arange(T, device=x.device).unsqueeze(0)
        distances = (indices - indices.T).abs().float().unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, T, T)
        sigma = self.sigma.abs() + 1e-6
        prior_attn = torch.exp(- (distances ** 2) / (2 * sigma ** 2))
        prior_attn = prior_attn / (prior_attn.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.matmul(series_attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_model)
        out = self.W_o(out)
        
        series_attn = series_attn + 1e-8
        prior_attn = prior_attn + 1e-8
        kl_loss = torch.mean(series_attn * (torch.log(series_attn) - torch.log(prior_attn)))
        return out, kl_loss

class HostAnomalyTransformer(nn.Module):
    def __init__(self, d_in, d_model=64, n_layers=3, n_heads=8):
        super().__init__()
        self.embedding = nn.Linear(d_in, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model))
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': AnomalyAttention(d_model, n_heads),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)),
                'norm2': nn.LayerNorm(d_model)
            }))
        self.recon = nn.Linear(d_model, d_in)

    def forward(self, X, mask=None, **kwargs):
        # X: [B, T, D]
        B, T, _ = X.shape
        x = self.embedding(X) + self.pos_enc[:, :T, :]
        total_assoc_loss = 0
        for layer in self.layers:
            attn_out, assoc_loss = layer['attn'](x)
            x = layer['norm1'](x + attn_out)
            total_assoc_loss += assoc_loss
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        x_hat = self.recon(x)
        recon_loss = ((X - x_hat) ** 2)
        if mask is not None:
            recon_loss = recon_loss * mask.unsqueeze(-1)
        recon_loss = recon_loss.mean(dim=-1).mean(dim=-1) # [B]
        return recon_loss.mean() + total_assoc_loss, recon_loss