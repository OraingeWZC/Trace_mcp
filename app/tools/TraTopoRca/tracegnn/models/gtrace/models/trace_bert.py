import torch
import torch.nn as nn

class BertLatencyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.Model.latency_feature_size
        
        base_in = config.Model.embedding_size * 3 + config.Latency.latency_feature_length
        if getattr(config.HostState, 'enable', False):
            base_in += config.HostState.out_dim
        if getattr(config.Model, 'enable_hetero', False) and getattr(config.HostState, 'enable', False):
            base_in += config.Model.host_topo_out_dim
            
        self.in_proj = nn.Linear(base_in, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, getattr(config.Model, 'bert_max_len', 100), d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=getattr(config.Model, 'bert_n_heads', 4),
            dim_feedforward=getattr(config.Model, 'bert_dim_feedforward', 256),
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=getattr(config.Model, 'bert_n_layers', 2))
        
        self.to_mu = nn.Linear(d_model, d_model)
        self.to_logvar = nn.Linear(d_model, d_model)

    def forward(self, padded_seq, mask):
        x = self.in_proj(padded_seq)
        B, T, _ = x.shape
        x = x + self.pos_encoder[:, :T, :]
        padding_mask = (mask == 0)
        h = self.transformer(x, src_key_padding_mask=padding_mask)
        
        mask_expanded = mask.unsqueeze(-1)
        h_pool = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-6)
        
        logvar = torch.tanh(self.to_logvar(h_pool))
        return self.to_mu(h_pool), logvar

class BertLatencyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.Model.latency_feature_size
        self.max_len = getattr(config.Model, 'bert_max_len', 100)
        
        self.z_to_seq = nn.Linear(d_model, d_model * self.max_len)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer_dec = nn.TransformerEncoder(decoder_layer, num_layers=2)
        
        self.out_mu = nn.Linear(d_model, config.Latency.latency_feature_length)
        self.out_logvar = nn.Linear(d_model, config.Latency.latency_feature_length)

    def forward(self, z):
        B = z.shape[0]
        x = self.z_to_seq(z).view(B, self.max_len, -1)
        x = self.transformer_dec(x)
        return self.out_mu(x), self.out_logvar(x)