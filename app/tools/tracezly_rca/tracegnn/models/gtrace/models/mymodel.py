# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import *
from tracegnn.models.gtrace.models.latency_embedding import latency_to_feature
from tracegnn.models.gtrace.models.tree_lstm import TreeLSTM
from tracegnn.models.gtrace.models.isoc_vgae import ISOC_VGAE
from tracegnn.models.gtrace.config import ExpConfig


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

    def forward(self, g: dgl.DGLGraph):
        operation = self.operation_embed(g.ndata['operation_id'])
        service   = self.service_embed(g.ndata['service_id'])
        status    = self.status_embed(g.ndata['status_id'])

        latency  = latency_to_feature(self.config, g.ndata['latency'], g.ndata['operation_id'], clip=False)
        features = torch.cat([operation, service, status, latency], dim=-1)
        return features


# =========================
# Tree-LSTM 延迟分支（保持你原接口/形状）
# =========================
class TreeLSTMLatencyEncoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(TreeLSTMLatencyEncoder, self).__init__()
        self.config = config

        self.linear_input = nn.Sequential(
                nn.Linear(config.Model.embedding_size * 3 + config.Latency.latency_feature_length, config.Model.latency_feature_size),
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

        self.structure_model = ISOC_VGAE(
            num_features=config.Model.num_features,
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

        assert n_z is None or n_z > 1
        n_z = 1 if n_z is None else n_z

        # ========== 0) 输入清洗，避免残留 NaN/Inf ==========
        self._sanitize_graph_inputs(g)

        # ========== 1) 节点嵌入 ==========
        embed = self.embedding(g)            # [N, 3*emb (+ L_latency_if_enabled)]
        # DEBUG
        self._assert_finite(embed, "embed")

        # ========== 2) 结构分支 ==========
        # IsoC-VGAE 内部完成自/邻域/度重构，返回结构损失与最后一层节点嵌入
        loss_structure, struct_emb = self.structure_model(adj, embed, degree, neighbor_dict)   # loss: scalar, struct_emb: [N, H]
        # DEBUG
        self._assert_finite(loss_structure, "loss_structure")

        # ========== 3) 延迟分支（变分）==========

        # 添加调试信息
        # print(f"Debug info - embed shape: {embed.shape}")

        # Encoder: z2 ~ q(z2|x)
        z_latency_mu, z_latency_logvar = self.latency_encoder(g, embed)  # [N, hid] each
        # DEBUG
        self._assert_finite(z_latency_logvar, "z_latency_logvar")

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
        # DEBUG
        self._assert_finite(latency_logvar, "latency_logvar")

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
        loss_latency, loss_latency_kl = latency_loss(self.config, pred_latency, g)
        # DEBUG
        self._assert_finite(loss_latency, "loss_latency_nll")
        self._assert_finite(loss_latency_kl, "loss_latency_kl")
        
        # 调试信息：打印各个损失组件
        # print(f"Debug - loss_latency (NLL): {loss_latency.item():.6f}")
        # print(f"Debug - loss_latency_kl: {loss_latency_kl.item():.6f}")
        # print(f"Debug - kl_weight: {self.config.Model.kl_weight}")
        # print(f"Debug - kl_weight * loss_latency_kl: {(self.config.Model.kl_weight * loss_latency_kl).item():.6f}")
        
        loss_latency = loss_latency + self.config.Model.kl_weight * loss_latency_kl
        
        # print(f"Debug - final loss_latency: {loss_latency.item():.6f}")

        # ========== 4) Kendall 不确定性加权 ==========
        # L_total = sum_i [ 1/(2*sigma_i^2) * L_i + log sigma_i ]
        # 恢复到原文件实现：不限制log_sigma和alpha/beta，让模型自由学习最优权重
        sigma_s = torch.exp(self.log_sigma_structure)
        sigma_l = torch.exp(self.log_sigma_latency)

        loss_total = (0.5 * (loss_structure / (sigma_s ** 2))
                      + 0.5 * (loss_latency   / (sigma_l ** 2))
                      + torch.log(sigma_s) + torch.log(sigma_l))
        # DEBUG
        self._assert_finite(loss_total, "loss_total")

        # 计算节点级别的异常分数
        # 结构异常分数基于IsoC-VGAE的重构误差和KL散度
        struct_scores = self._compute_node_structure_scores(g, adj, embed, degree, neighbor_dict)
        
        # 延迟异常分数基于节点延迟重构误差
        latency_scores = self._compute_node_latency_scores(g, pred_latency)
        
        return {
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
            "alpha": 0.5 / (sigma_s ** 2),           # 结构异常权重（无限制，原文件实现）
            "beta": 0.5 / (sigma_l ** 2)             # 延迟异常权重（无限制，原文件实现）
        }

    # DEBUG
    @staticmethod
    def _assert_finite(tensor: torch.Tensor, name: str):
        """
        快速定位 NaN/Inf。发现异常立即报错，中断当前迭代。
        """
        if not torch.isfinite(tensor).all():
            raise ValueError(f"[NaN Debug] {name} contains non-finite values.")

    def _sanitize_graph_inputs(self, g: dgl.DGLGraph):
        """
        对图的节点特征做一次防御性清洗，避免 NaN/Inf/越界 ID。
        """
        # 连续值：latency
        if 'latency' in g.ndata:
            g.ndata['latency'] = torch.nan_to_num(
                g.ndata['latency'],
                nan=0.0,
                posinf=1e6,
                neginf=0.0,
            )
        # 离散 ID：operation/service/status
        for key in ('operation_id', 'service_id', 'status_id'):
            if key in g.ndata:
                tensor = g.ndata[key]
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                tensor = torch.clamp(tensor, min=0)
                g.ndata[key] = tensor.long()
        # 运行时的 latency_range 也防御一次，避免均值/方差为 NaN 或过小
        if hasattr(self.config, "RuntimeInfo") and getattr(self.config.RuntimeInfo, "latency_range", None) is not None:
            lr = self.config.RuntimeInfo.latency_range
            lr = torch.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
            # std 位于第 1 列，做下界限制
            if lr.shape[1] >= 2:
                lr[:, 1] = torch.clamp(lr[:, 1], min=1e-3)
            self.config.RuntimeInfo.latency_range = lr

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
            
            # NEW：应用"Reducing the Entropy Gap"优化原则：放大结构得分
            struct_scores = struct_scores * 3
            
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
        
        # NEW：应用"Reducing the Entropy Gap"优化原则：限制标准差
        # 从logvar计算标准差
        sigma = torch.exp(0.5 * pred_latency['latency_logvar'])
        
        # 使用固定的阈值或基于现有配置动态计算阈值
        # 方法1: 使用固定的p99阈值（例如，取logvar的99%分位数作为阈值）
        # 方法2: 基于当前批次数据动态计算一个合理的阈值
        if sigma.numel() > 0:  # 确保张量不为空
            # 计算当前sigma的99%分位数作为阈值
            sigma_p99_threshold = torch.quantile(sigma.flatten(), 0.99, dim=0)
            # 为每个节点应用阈值限制
            sigma_limited = torch.min(sigma, sigma_p99_threshold.expand_as(sigma))
            # 使用受限标准差重新计算NLL
            logvar_limited = 2 * torch.log(sigma_limited)
            
            # 重新计算受限的延迟分数
            limited_latency_scores = normal_loss(
                latency_label,
                pred_latency['latency_mu'],
                logvar_limited,
                reduction='none'
            )
            # 对特征维度求平均
            latency_scores = torch.mean(limited_latency_scores, dim=1)  # [N]
        
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
    恢复到原文件实现：简单直接，不添加额外的clamp限制
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
    # 恢复到原文件实现：简单直接
    return -0.5 * torch.mean(logvar + 1. - torch.exp(logvar) - mu ** 2)


def latency_loss(config: ExpConfig, pred: dict, graphs: dgl.DGLGraph, clip: bool=True, return_detail: bool=False) -> torch.Tensor:
    graph_list: List[dgl.DGLGraph] = dgl.unbatch(graphs)
    device = graphs.device

    b = len(graph_list)

    latency_label = torch.zeros(
        [b, config.decoder_max_nodes, config.Latency.latency_feature_length], dtype=torch.float, device=device)
    latency_label = latency_to_feature(config, graphs.ndata['latency'], graphs.ndata['operation_id'], clip=clip)
    
    operation_loss_dict = torch.zeros([config.DatasetParams.operation_cnt], dtype=torch.float, device=config.device)
    operation_cnt_dict = torch.zeros([config.DatasetParams.operation_cnt], dtype=torch.long, device=config.device)

    if config.Latency.embedding_type != 'class':
        latency_loss = normal_loss(latency_label, pred['latency_mu'], pred['latency_logvar'], reduction='none')
        operation_id = graphs.ndata['operation_id']
        latency_loss = torch.mean(latency_loss, dim=1)
        operation_cnt_dict = torch.index_add(operation_cnt_dict, 0, operation_id, torch.ones_like(operation_id))
        operation_loss_dict = torch.index_add(operation_loss_dict, 0, operation_id, latency_loss)
        latency_loss = latency_loss.mean()
    else:
        latency_loss = F.cross_entropy(pred['latency_mu'], latency_label)
    
    if config.Model.vae:
        kl_latency = kl_loss(pred['z_latency_mu'], pred['z_latency_logvar'])
    else:
        kl_latency = torch.tensor(0.0, device=config.device)
    
    if return_detail:
        return latency_loss, kl_latency, operation_loss_dict, operation_cnt_dict
    else:
        return latency_loss, kl_latency


# 邻接矩阵 -> neighbor_dict（用于 IsoC-VGAE）
def construct_neighbor_dict(adj):
    neighbor_dict = {x: [] for x in range(adj.size(0))}
    source_nodes, target_nodes, _ = adj.coo()
    edges = torch.cat([torch.reshape(source_nodes, [-1, 1]), torch.reshape(target_nodes, [-1, 1])], 1)
    for source_node, target_node in edges:
        neighbor_dict[source_node.item()].append(target_node.item())
    return neighbor_dict


