# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import *
from tracegnn.models.gtrace.models.latency_embedding import latency_to_feature
from tracegnn.models.gtrace.models.tree_lstm import TreeLSTM
from tracegnn.models.gtrace.models.isoc_vgae import ISOC_VGAE
from tracegnn.models.gtrace.config import ExpConfig


class HostVAE(nn.Module):
    """
    Simple MLP-VAE for host_state vectors.
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
    """Simplified OmniAnomaly (GRU + VAE) operating on sequences X:[B,T,D]."""
    def __init__(self, d_in: int, h: int, z: int):
        super().__init__()
        self.enc = nn.GRU(input_size=d_in, hidden_size=h, batch_first=True)
        self.mu_z = nn.Linear(h, z)
        self.logvar_z = nn.Linear(h, z)
        self.dec = nn.GRU(input_size=z, hidden_size=h, batch_first=True)
        self.mu_x = nn.Linear(h, d_in)
        self.logvar_x = nn.Linear(h, d_in)

    def forward(self, X: torch.Tensor, beta_kl: float, mask: Optional[torch.Tensor] = None):
        # X: [B, T, D], mask: [B, T] (1=valid)
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


# =========================
# 节点离散属性嵌入（保持你原实现）
# =========================
class LevelEmbedding(nn.Module):
    def __init__(self, config: ExpConfig):
        super(LevelEmbedding, self).__init__()
        self.config = config

        self.operation_embed = nn.Embedding(config.DatasetParams.operation_cnt, config.Model.embedding_size)
        self.service_embed   = nn.Embedding(config.DatasetParams.service_cnt,   config.Model.embedding_size)
        self.status_embed    = nn.Embedding(config.DatasetParams.status_cnt,    config.Model.embedding_size)
        # HostTopo encoder (Phase 3)
        if getattr(config.Model, 'enable_hetero', False) and getattr(config.HostState, 'enable', False):
            # in_dim equals HostState in_dim after projection; reuse HostState out_dim as base
            self.host_topo_proj = nn.Sequential(
                nn.Linear(getattr(config.HostState, 'per_metric_dims', 3) * max(1, len(getattr(config.HostState, 'metrics', []))),
                          config.Model.host_topo_out_dim),
                nn.ReLU(),
            )
        # Host state encoder (Phase 2)
        if getattr(config.HostState, 'enable', False):
            in_dim = int(getattr(config.HostState, 'per_metric_dims', 3)) * max(1, len(getattr(config.HostState, 'metrics', [])))
            self.host_state_proj = nn.Sequential(
                nn.Linear(in_dim, config.HostState.out_dim),
                nn.ReLU(),
            )

    def forward(self, g: dgl.DGLGraph):
        operation = self.operation_embed(g.ndata['operation_id'])
        service   = self.service_embed(g.ndata['service_id'])
        status    = self.status_embed(g.ndata['status_id'])

        latency  = latency_to_feature(self.config, g.ndata['latency'], g.ndata['operation_id'], clip=False)
        feats = [operation, service, status, latency]
        if getattr(self.config.HostState, 'enable', False) and ('host_state' in g.ndata):
            host_state = g.ndata['host_state'].float()
            try:
                host_feat = self.host_state_proj(host_state)
                feats.append(host_feat)
            except Exception:
                pass
        if getattr(self.config.Model, 'enable_hetero', False) and getattr(self.config.HostState, 'enable', False) and ('host_topo_agg' in g.ndata):
            topo = g.ndata['host_topo_agg'].float()
            try:
                topo_feat = self.host_topo_proj(topo)
                feats.append(topo_feat)
            except Exception:
                pass
        features = torch.cat(feats, dim=-1)
        return features


# =========================
# Tree-LSTM 延迟分支（保持你原接口/形状）
# =========================
class TreeLSTMLatencyEncoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(TreeLSTMLatencyEncoder, self).__init__()
        self.config = config

        base_in = config.Model.embedding_size * 3 + config.Latency.latency_feature_length
        if getattr(config.HostState, 'enable', False):
            base_in += config.HostState.out_dim
        if getattr(config.Model, 'enable_hetero', False) and getattr(config.HostState, 'enable', False):
            base_in += config.Model.host_topo_out_dim
        self.linear_input = nn.Sequential(
                nn.Linear(base_in, config.Model.latency_feature_size),
                nn.ReLU(),
                nn.Linear(config.Model.latency_feature_size, config.Model.latency_feature_size * 2)
            )

    def forward(self, g: dgl.DGLGraph, embed: torch.Tensor):
        y = self.linear_input(embed)  # [N, 2*hid]
        z_latency_mu, z_latency_logvar = torch.split(y, y.size(-1) // 2, dim=-1)
        z_latency_logvar = torch.tanh(z_latency_logvar)  # 稳定训练
        return z_latency_mu, z_latency_logvar


class TreeLSTMLatencyDecoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(TreeLSTMLatencyDecoder, self).__init__()
        self.config = config

        self.tree_lstm = TreeLSTM(config.Model.latency_feature_size, config.Model.latency_feature_size)

        # Mu & Sigma
        # 创建两个线性层，一个用于输出延迟的均值，一个用于输出延迟的方差
        self.linear_output_mu = nn.Linear(config.Model.latency_feature_size, config.Latency.latency_feature_length)
        self.linear_output_logvar = nn.Linear(config.Model.latency_feature_size, config.Latency.latency_feature_length)

        # Add op-wise output layer
        # 创建一个嵌入层，用于用于存储每种操作类型的操作特定权重
        self.op_wise_output = nn.Embedding(
            num_embeddings=config.DatasetParams.operation_cnt + 1,
            embedding_dim=2 * config.Latency.latency_feature_length ** 2
        )

    def forward(self, g: dgl.DGLGraph, z_latency: torch.Tensor):
        y = self.tree_lstm(g, z_latency)

        mu = self.linear_output_mu(y)
        logvar = self.linear_output_logvar(y)

        w_mu, w_logvar = torch.split(
            self.op_wise_output(g.ndata['operation_id']).reshape(-1, 1, self.config.Latency.latency_feature_length,
                self.config.Latency.latency_feature_length * 2),
            split_size_or_sections=self.config.Latency.latency_feature_length,
            dim=-1
        )

        mu = (mu.unsqueeze(2) @ (w_mu)).squeeze(2)
        logvar = (logvar.unsqueeze(2) @ (w_logvar)).squeeze(2)

        return mu, logvar

# =========================
# 总模型：IsoC-VGAE(结构) + Tree-LSTM(延迟)
# =========================
class MyTraceAnomalyModel(nn.Module):
    """
    结构分支：IsoC-VGAE -> 返回结构重构损失 + 结构嵌入(struct_emb)
    延迟分支：Tree-LSTM 变分建模 -> 返回 latency_mu/logvar，并以 NLL+KL 作为延迟损失
    联合优化：Kendall 不确定性加权（learnable log_sigma_*）
    """
    def __init__(self, config: 'ExpConfig'):
        super(MyTraceAnomalyModel, self).__init__()
        self.config = config
        self.device = config.device

        # Structure Model
        self.embedding = LevelEmbedding(config)

        # Effective feature dim for structure branch
        eff_num_features = config.Model.embedding_size * 3 + config.Latency.latency_feature_length
        if getattr(config.HostState, 'enable', False):
            eff_num_features += config.HostState.out_dim
        if getattr(config.Model, 'enable_hetero', False) and getattr(config.HostState, 'enable', False):
            eff_num_features += config.Model.host_topo_out_dim
        self.structure_model = ISOC_VGAE(
            num_features=eff_num_features,
            hidden_dim=config.Model.hidden_dim,
            device=config.device,
            GNN_name=getattr(config.Model, 'structure_gnn', 'GIN'),
            dropout=getattr(config.Model, 'dropout', 0.0),
            bn='BatchNorm1d'
        )

        # Latency Model
        self.latency_encoder = TreeLSTMLatencyEncoder(config)
        self.latency_decoder = TreeLSTMLatencyDecoder(config)

        # ===== Kendall 不确定性参数 =====
        # 注意：Kendall 论文中是 1/(2*sigma^2)*L + log sigma，这里学习 log_sigma
        self.log_sigma_structure = nn.Parameter(torch.zeros(1))
        self.log_sigma_latency   = nn.Parameter(torch.zeros(1))

        # Host-Channel Kendall head (optional)
        hc_cfg = getattr(self.config, "HostChannel", None)
        self.enable_host_channel = bool(
            getattr(hc_cfg, "enable", False)
            and getattr(self.config.HostState, "enable", False)
        )
        # backend and KL weight for host head
        self.host_backend = str(getattr(hc_cfg, 'backend', 'omni')) if hc_cfg is not None else 'omni'
        self.beta_host_kl = float(getattr(hc_cfg, 'beta_kl', 1e-3)) if hc_cfg is not None else 1e-3

        if self.enable_host_channel:
            in_dim = int(getattr(self.config.HostState, "per_metric_dims", 3)) * max(
                1, len(getattr(self.config.HostState, "metrics", []))
            )
            hidden_dim = int(getattr(hc_cfg, "hidden_dim", 64))
            latent_dim = int(getattr(hc_cfg, "latent_dim", 16))
            self.host_vae = HostVAE(in_dim, hidden_dim, latent_dim)
            init_logvar = float(getattr(hc_cfg, "kendall_init_logvar", 0.0))
            self.log_sigma_host = nn.Parameter(torch.tensor(init_logvar))
        else:
            self.host_vae = None
            self.log_sigma_host = None

        # Optional OmniAnomaly host sequence head
        if self.enable_host_channel and (self.host_backend == 'omni'):
            try:
                D = len(getattr(hc_cfg, 'seq_metrics', ['cpu', 'mem', 'fs']))
            except Exception:
                D = 3
            hdim = int(getattr(hc_cfg, 'hidden_dim', 64)) if hc_cfg is not None else 64
            zdim = int(getattr(hc_cfg, 'latent_dim', 16)) if hc_cfg is not None else 16
            self.host_omni = HostOmniAnomaly(d_in=D, h=hdim, z=zdim)
        else:
            self.host_omni = None

    def sample_z(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        # Sample eps from N (0, I)
        eps = torch.randn_like(z_mu)
        sigma = torch.exp(z_logvar * 0.5)

        result = z_mu + sigma * eps
        return result

    def forward(self,
                g: dgl.DGLGraph,
                adj: torch.Tensor,
                degree: torch.Tensor,
                neighbor_dict: Dict[int, List[int]],
                n_z: int=None):
        """
        参数:
            g: DGLGraph，含结点字段 ['operation_id','service_id','status_id','latency', ...]
            adj: 邻接矩阵 (torch.FloatTensor) 形如 [N, N]
            degree: 节点度 (torch.Long/FloatTensor) [N]
            neighbor_dict: 字典 {node_id: [neighbor_ids]}
            n_z: 重要性采样数（用于 VAE Monte-Carlo），默认取 config.Model.n_z

        返回:
            dict，含：
            - loss_total, loss_structure, loss_latency
            - struct_emb: 结构分支最终图/节点嵌入（取 self.structure_model 的最后层输出）
            - latency_mu/logvar: 延迟分支的重构分布参数
        """
        device = self.device
        # 评估/训练稳定性：限制不确定性权重的范围，避免极端漂移
        try:
            self.log_sigma_structure.data.clamp_(-2.0, 2.0)
            self.log_sigma_latency.data.clamp_(-2.0, 2.0)
            if self.log_sigma_host is not None:
                self.log_sigma_host.data.clamp_(-2.0, 2.0)
        except Exception:
            pass

        assert n_z is None or n_z > 1
        n_z = 1 if n_z is None else n_z

        # ========== 1) 节点嵌入 ==========
        embed = self.embedding(g)            # [N, 3*emb (+ L_latency_if_enabled)]

        # ========== 2) 结构分支 ==========
        # IsoC-VGAE 内部完成自/邻域/度重构，返回结构损失与最后一层节点嵌入
        loss_structure, struct_emb = self.structure_model(adj, embed, degree, neighbor_dict)   # loss: scalar, struct_emb: [N, H]

        # ========== 3) 延迟分支（变分）==========

        # 添加调试信息
        # print(f"Debug info - embed shape: {embed.shape}")

        # Encoder: z2 ~ q(z2|x)
        z_latency_mu, z_latency_logvar = self.latency_encoder(g, embed)  # [N, hid] each

        # 添加调试信息
        # print("Encoder Debug")
        # print(f"Debug info - z_latency_mu shape: {z_latency_mu.shape}")
        # print(f"Debug info - z_latency_logvar shape: {z_latency_logvar.shape}")

        z_latency_mu     = z_latency_mu.unsqueeze(1).expand(-1, n_z, -1)       # [N, n_z, hid]
        z_latency_logvar = z_latency_logvar.unsqueeze(1).expand(-1, n_z, -1)   # [N, n_z, hid]

        # 采样 z2
        z_latency_sample = self.sample_z(z_latency_mu, z_latency_logvar)     # [N, (n_z or 1), hid]

        # 添加调试信息
        # print(f"Debug info - z_latency_sample shape: {z_latency_sample.shape}")

        # Decoder: p(latency|z2)
        latency_mu, latency_logvar = self.latency_decoder(g, z_latency_sample)      # [N, (n_z or 1), L]

        # 添加调试信息
        # print(f"Debug info - latency_mu shape: {latency_mu.shape}")
        # print(f"Debug info - latency_logvar shape: {latency_logvar.shape}")

        # 统一形状（n_z==1 时 squeeze 成 [N, L]，保留 loss 里对两种形状的兼容）
        if n_z == 1:
            z_latency_mu, z_latency_logvar = z_latency_mu.squeeze(1), z_latency_logvar.squeeze(1)     # [N, hid]
            z_latency_sample                = z_latency_sample.squeeze(1)                              # [N, hid]
            latency_mu, latency_logvar      = latency_mu.squeeze(1), latency_logvar.squeeze(1)        # [N, L]
        else:
            # 当 n_z > 1 时，对所有输出在 n_z 维度上求平均，保持与 n_z=1 时相同的形状
            latency_mu = torch.mean(latency_mu, dim=1)        # [N, L]
            latency_logvar = torch.mean(latency_logvar, dim=1)  # [N, L]
            z_latency_mu = torch.mean(z_latency_mu, dim=1)    # [N, hid]
            z_latency_logvar = torch.mean(z_latency_logvar, dim=1)  # [N, hid]

        # 组装 pred 喂给延迟损失
        pred_latency = {
            'latency_mu': latency_mu,
            'latency_logvar': latency_logvar,
            'z_latency_mu': z_latency_mu,
            'z_latency_logvar': z_latency_logvar
        }
        # Use CPU graph for latency loss graph ops to avoid GPU DGL kernel issues
        g_cpu = g.to('cpu') if 'cuda' in str(self.config.device) else g
        loss_latency, loss_latency_kl = latency_loss(self.config, pred_latency, g_cpu)
        loss_latency = loss_latency + self.config.Model.kl_weight * loss_latency_kl

        # Host-Channel auxiliary head: prefer OmniAnomaly on sequences if available; fallback to snapshot VAE
        loss_host = torch.tensor(0.0, device=device)
        host_nll_dict: Optional[Dict[int, float]] = None
        if getattr(self, "enable_host_channel", False) and ("host_id" in g.ndata):
            # Try sequence backend first
            seq_dict = getattr(g, 'host_seq', None)
            if (self.host_backend == 'omni') and (seq_dict is not None) and isinstance(seq_dict, dict) and (self.host_omni is not None) and len(seq_dict) > 0:
                try:
                    # sort by host id for stable stacking
                    hids, seqs = zip(*sorted(seq_dict.items(), key=lambda kv: int(kv[0])))
                    X = torch.stack([s.to(device) for s in seqs], dim=0)  # [B,T,D]
                    mask = torch.ones((X.shape[0], X.shape[1]), device=device)
                    L_host, nll_per_host = self.host_omni(X, beta_kl=self.beta_host_kl, mask=mask)
                    loss_host = L_host
                    host_nll_dict = {int(h): float(v.detach().item()) for h, v in zip(hids, nll_per_host)}
                except Exception:
                    pass
            # Fallback to snapshot host_state VAE
            if (host_nll_dict is None) and ("host_state" in g.ndata) and (getattr(self, 'host_vae', None) is not None):
                hs = g.ndata["host_state"].float()
                hids = g.ndata["host_id"].long()
                uniq = torch.unique(hids)
                host_vecs: List[torch.Tensor] = []
                host_keys: List[int] = []
                for hid in uniq.tolist():
                    if hid <= 0:
                        continue
                    mask = (hids == hid)
                    if not bool(mask.any()):
                        continue
                    host_vecs.append(hs[mask].mean(dim=0, keepdim=True))
                    host_keys.append(int(hid))
                if host_vecs:
                    X = torch.cat(host_vecs, dim=0)
                    out_h = self.host_vae(X)
                    beta_h = float(getattr(self.config.HostChannel, "beta_kl", 1e-3))
                    host_nll = out_h["recon"] + beta_h * out_h["kl"]
                    loss_host = host_nll.mean()
                    host_nll_dict = {hk: float(v.detach().item()) for hk, v in zip(host_keys, host_nll)}

        # ========== 4) Kendall 不确定性加权 ==========
        # L_total = sum_i [ 1/(2*sigma_i^2) * L_i + log sigma_i ]
        sigma_s = torch.exp(self.log_sigma_structure)
        sigma_l = torch.exp(self.log_sigma_latency)

        loss_total = (0.5 * (loss_structure / (sigma_s ** 2))
                      + 0.5 * (loss_latency   / (sigma_l ** 2))
                      + torch.log(sigma_s) + torch.log(sigma_l))

        if getattr(self, "enable_host_channel", False) and (self.log_sigma_host is not None):
            sigma_h = torch.exp(self.log_sigma_host)
            loss_total = loss_total + 0.5 * (loss_host / (sigma_h ** 2)) + torch.log(sigma_h)

        # 计算节点级别的异常分数
        # 结构异常分数基于IsoC-VGAE的重构误差和KL散度
        struct_scores = self._compute_node_structure_scores(g, adj, embed, degree, neighbor_dict)
        
        # 延迟异常分数基于节点延迟重构误差
        latency_scores = self._compute_node_latency_scores(g, pred_latency)

        result = {
            "loss_total":       loss_total,
            "loss_structure":   loss_structure.detach(),
            "loss_latency":     loss_latency.detach(),
            "loss_latency_kl":    loss_latency_kl.detach(),
            "struct_emb":       struct_emb,          # [N, H]
            "latency_mu":       latency_mu,          # [N, L] 或 [N, n_z, L]
            "latency_logvar":   latency_logvar,      # 同上
            "z_latency_mu":     z_latency_mu,        # [N, hid] 或 [N, n_z, hid]
            "z_latency_logvar": z_latency_logvar,     # 同上
            "node_structure_scores": struct_scores,  # 节点结构异常分数 [N]
            "node_latency_scores": latency_scores,   # 节点延迟异常分数 [N]
            "alpha": 0.5 / (sigma_s ** 2),           # 结构异常权重
            "beta": 0.5 / (sigma_l ** 2)             # 延迟异常权重
        }
        # 暴露主机 Kendall 权重 gamma（用于评估端融合权重）
        if getattr(self, "enable_host_channel", False) and (self.log_sigma_host is not None):
            try:
                sigma_h = torch.exp(self.log_sigma_host)
                result["gamma"] = 0.5 / (sigma_h ** 2)
            except Exception:
                pass
        if host_nll_dict is not None:
            try:
                result['loss_host'] = loss_host.detach()
            except Exception:
                result['loss_host'] = loss_host
            result['host_nll_dict'] = host_nll_dict

        return result

    def _compute_node_structure_scores(self, g, adj, embed, degree, neighbor_dict):
        """
        计算每个节点的结构异常分数
        基于IsoC-VGAE的边重构误差和节点潜在向量KL散度
        """
        # 获取结构模型的详细输出
        with torch.no_grad():
            # 这里我们重新运行结构模型以获取详细信息
            h_list = self.structure_model.encoder(embed, adj)
            
            # 计算自重构误差 (节点特征重构)
            self_recon_loss = []
            for layer in range(self.structure_model.num_layers - 1, -1, -1):
                if layer == 0:
                    target_embedding = embed
                else:
                    target_embedding = h_list[layer - 1]
                
                mean = self.structure_model.reconstruct_self[self.structure_model.num_layers - layer - 1](h_list[layer])
                if layer < self.structure_model.num_layers - 1:
                    mean_prior = self.structure_model.decode_mean[self.structure_model.num_layers - layer - 2](mean_prior)
                else:
                    mean_prior = torch.zeros_like(mean)


                mean_posterior = mean + mean_prior
                recon_error = torch.mean((target_embedding - mean_posterior) ** 2, dim=1)  # [N]
                self_recon_loss.append(recon_error)
            
            # 平均各层的自重构误差
            avg_self_recon = torch.mean(torch.stack(self_recon_loss), dim=0)  # [N]
            
            # 计算邻居分布KL散度
            neighbor_kl_div = []
            for layer in range(self.structure_model.num_layers - 1, -1, -1):
                if layer == 0:
                    target_embedding = embed
                else:
                    target_embedding = h_list[layer - 1]
                
                mean = self.structure_model.reconstruct_self[self.structure_model.num_layers - layer - 1](h_list[layer])
                if layer < self.structure_model.num_layers - 1:
                    mean_prior = self.structure_model.decode_mean[self.structure_model.num_layers - layer - 2](mean_prior)
                else:
                    mean_prior = torch.zeros_like(mean)
                
                mean_posterior = mean + mean_prior
                log_std = self.structure_model.decode_std[self.structure_model.num_layers - layer - 1](h_list[layer])
                
                # 计算邻域分布
                neighbor_mean = self._neighborhood_distribution(target_embedding, neighbor_dict)
                kl = 0.5 * torch.sum(-1 - 2 * log_std + (mean_posterior - neighbor_mean) ** 2 + torch.exp(2 * log_std), dim=1)
                neighbor_kl_div.append(kl)
            
            # 平均各层的KL散度
            avg_neighbor_kl = torch.mean(torch.stack(neighbor_kl_div), dim=0)  # [N]
            
            # 组合结构分数
            struct_scores = (self.structure_model.lambda_self * avg_self_recon + 
                           self.structure_model.lambda_neighbor * avg_neighbor_kl)
            
        return struct_scores

    def _compute_node_latency_scores(self, g, pred_latency):
        """
        计算每个节点的延迟异常分数
        基于节点的时间序列重建误差
        """
        # 获取延迟标签
        latency_label = latency_to_feature(
            self.config, g.ndata['latency'], g.ndata['operation_id'], clip=False
        )
        
        # 计算每个节点的延迟重构误差
        latency_scores = normal_loss(
            latency_label, 
            pred_latency['latency_mu'], 
            pred_latency['latency_logvar'], 
            reduction='none'
        )
        
        # 对特征维度求平均
        latency_scores = torch.mean(latency_scores, dim=1)  # [N]
        
        return latency_scores

    def _neighborhood_distribution(self, embedding, neighbor_dict):
        """
        计算节点邻域的平均分布（更健壮的实现）
        - embedding: tensor [N, D]
        - neighbor_dict: mapping node_id -> iterable(neighbor_ids)
        (支持 0-based 或 1-based 索引，支持 list / tensor / str keys 等)
        返回 mean: tensor [N, D]
        """
        device = embedding.device
        N, D = embedding.shape
        mean = torch.empty_like(embedding, device=device)

        # 先把 neighbor_dict 标准化为每个 0..N-1 节点对应的邻居列表（Python list of ints）
        normalized = [[] for _ in range(N)]

        for raw_k, raw_neighbors in neighbor_dict.items():
            # 试着把 key 转成 int（有可能是 str '0' / int 1-based / int 0-based）
            try:
                k = int(raw_k)
            except Exception:
                # 如果 key 不能被转成 int，跳过
                continue

            # 处理 key 可能是 1-based 的情况：如果 k 在 [1, N] 而不是 [0, N-1]，把它转换为 0-based
            if 0 <= k < N:
                node_idx = k
            elif 1 <= k <= N:
                node_idx = k - 1
            else:
                # 不在合理范围，跳过
                continue

            # 解析 neighbors（支持 tensor、list、set 等）
            neighs_list = []
            if raw_neighbors is None:
                neighs_list = []
            else:
                if isinstance(raw_neighbors, torch.Tensor):
                    try:
                        neighs_list = raw_neighbors.long().tolist()
                    except Exception:
                        neighs_list = []
                else:
                    # 如果是可迭代对象（list/tuple/set），尝试逐项转 int
                    try:
                        neighs_list = [int(x) for x in raw_neighbors]
                    except Exception:
                        # 不能解析则跳过该条目
                        neighs_list = []

            # 如果邻居索引看起来是 1-based（存在 >= N），把它们减 1
            if neighs_list and max(neighs_list) >= N:
                neighs_list = [n - 1 for n in neighs_list]

            # 过滤越界和重复
            neighs_list = [n for n in neighs_list if 0 <= n < N]
            # 可选：去重并保持顺序
            seen = set()
            filtered = []
            for n in neighs_list:
                if n not in seen:
                    seen.add(n)
                    filtered.append(n)

            normalized[node_idx] = filtered

        # 现在为每个节点计算邻域均值：若没有有效邻居，用自身 embedding（避免置零）
        for i in range(N):
            neighs = normalized[i]
            if neighs:
                idx_tensor = torch.tensor(neighs, dtype=torch.long, device=device)
                # 若只有一个邻居，mean 仍然工作
                mean[i, :] = torch.mean(embedding[idx_tensor, :], dim=0)
            else:
                # 没有邻居或缺少 key -> 使用自身 embedding 作为邻域均值（这样不会人工放大差值）
                mean[i, :] = embedding[i, :]

        return mean




# =========================
# 计算损失的函数
# =========================
def normal_loss(label: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor,
                reduction: str = 'mean',
                eps: float = 1e-7,
                positive_only: bool = False) -> torch.Tensor:
    """
    高斯负对数似然（NLL）
    """
    if positive_only:
        loss_mask = (label > mu)
        loss = (mu - label) ** 2 / (2 * torch.exp(logvar) + eps) * loss_mask + 0.5 * logvar
    else:
        loss = (mu - label) ** 2 / (2 * torch.exp(logvar) + eps) + 0.5 * logvar

    if reduction == 'mean':
        return torch.mean(loss)
    else:
        return loss


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # 对节点/样本做均值
    return -0.5 * torch.mean(logvar + 1. - torch.exp(logvar) - mu ** 2)


def latency_loss(config: ExpConfig, pred: dict, graphs: dgl.DGLGraph, clip: bool=True, return_detail: bool=False) -> torch.Tensor:
    """Compute latency loss using CPU graph for graph ops and device tensors for math.
    graphs: DGLGraph on CPU (caller ensures); tensors computed on config.device.
    """
    device = config.device

    # Build labels on device from CPU graph node data
    lat_cpu = graphs.ndata['latency']
    opid_cpu = graphs.ndata['operation_id']
    latency_label = latency_to_feature(
        config,
        lat_cpu.to(device),
        opid_cpu.to(device),
        clip=clip
    )

    operation_loss_dict = torch.zeros([config.DatasetParams.operation_cnt], dtype=torch.float, device=device)
    operation_cnt_dict  = torch.zeros([config.DatasetParams.operation_cnt], dtype=torch.long,  device=device)

    if config.Latency.embedding_type != 'class':
        lat_loss_all = normal_loss(latency_label, pred['latency_mu'], pred['latency_logvar'], reduction='none')
        lat_loss_node = torch.mean(lat_loss_all, dim=1)  # mean over feature dim
        opid_dev = opid_cpu.to(device)
        ones = torch.ones_like(opid_dev, dtype=torch.long, device=device)
        operation_cnt_dict = torch.index_add(operation_cnt_dict, 0, opid_dev, ones)
        operation_loss_dict = torch.index_add(operation_loss_dict, 0, opid_dev, lat_loss_node)
        latency_loss_val = lat_loss_node.mean()
    else:
        latency_loss_val = F.cross_entropy(pred['latency_mu'], latency_label)

    if config.Model.vae:
        kl_latency = kl_loss(pred['z_latency_mu'], pred['z_latency_logvar'])
    else:
        kl_latency = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    if return_detail:
        return latency_loss_val, kl_latency, operation_loss_dict, operation_cnt_dict
    else:
        return latency_loss_val, kl_latency


# 邻接矩阵 -> neighbor_dict（用于 IsoC-VGAE）
def construct_neighbor_dict(adj_or_edge_index):
    """Build neighbor dict from SparseTensor or edge_index (2xE tensor)."""
    # SparseTensor path
    if hasattr(adj_or_edge_index, 'coo'):
        adj = adj_or_edge_index
        neighbor_dict = {x: [] for x in range(adj.size(0))}
        source_nodes, target_nodes, _ = adj.coo()
        edges = torch.cat([torch.reshape(source_nodes, [-1, 1]), torch.reshape(target_nodes, [-1, 1])], 1)
        for source_node, target_node in edges:
            neighbor_dict[source_node.item()].append(target_node.item())
        return neighbor_dict
    # edge_index path (optionally with total N): accepts Tensor or (Tensor, N)
    if isinstance(adj_or_edge_index, tuple) and len(adj_or_edge_index) == 2:
        edge_index, total_N = adj_or_edge_index
    else:
        edge_index, total_N = adj_or_edge_index, None

    if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 2:
        if edge_index.numel() == 0:
            N = int(total_N) if total_N is not None else 0
            return {i: [] for i in range(N)}
        max_idx = int(max(edge_index[0].max().item(), edge_index[1].max().item())) if edge_index.numel() > 0 else -1
        N = int(total_N) if total_N is not None else (max_idx + 1)
        neighbor_dict = {i: [] for i in range(N)}
        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()
        for s, t in zip(src, dst):
            if 0 <= s < N and 0 <= t < N:
                neighbor_dict[int(s)].append(int(t))
        return neighbor_dict
    # fallback empty
    return {}
