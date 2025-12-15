from tracegnn.models.gtrace.models.layers import MLP, GCNLayer, GINLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

class ISOC_VGAE(nn.Module):
    def __init__(self, num_features, hidden_dim, device, GNN_name = "GIN", 
                 dropout = 0., bn = 'BatchNorm1d', lambda_self = 1, lambda_neighbor = 0.0001, lambda_degree = 10):
        super(ISOC_VGAE, self).__init__()
        self.input_dim = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)
        self.lambda_self = lambda_self
        self.lambda_neighbor = lambda_neighbor
        self.lambda_degree = lambda_degree
        self.device = device
        
        GNNEncoder = []
        decode_mean = []
        decode_std = []
        reconstruct_self = []
        reconstruct_degree = []
        input_dim = num_features
        for layer in range(self.num_layers):
            # encoder layers
            if GNN_name == "GIN":
                GNNEncoder.append(GINLayer(input_dim = input_dim, 
                                           hidden_dim = hidden_dim[layer], 
                                           output_dim = hidden_dim[layer], 
                                           num_hidden = 2,
                                           act = lambda x: x if layer == self.num_layers - 1 else nn.ReLU(),
                                           dropout = 0. if layer == 0 else dropout,
                                           bn = None if layer == self.num_layers - 1 else bn))
            else:
                GNNEncoder.append(GCNLayer(input_dim = input_dim, 
                                           output_dim = hidden_dim[layer], 
                                           act = lambda x: x if layer == self.num_layers - 1 else nn.ReLU(), 
                                           dropout = 0. if layer == 0 else dropout,
                                           bn = None if layer == self.num_layers - 1 else bn))
            input_dim = hidden_dim[layer]
            # decoder layers
            if layer == self.num_layers - 1:
                output_dim = num_features
            else:
                output_dim = hidden_dim[self.num_layers - layer - 2]
            if layer > 0:
                decode_mean.append(MLP(hidden_dim[self.num_layers - layer - 1], hidden_dim[self.num_layers - layer - 1], output_dim, num_hidden = 2))
            decode_std.append(MLP(hidden_dim[self.num_layers - layer - 1], hidden_dim[self.num_layers - layer - 1], output_dim, num_hidden = 2))
            reconstruct_self.append(MLP(hidden_dim[self.num_layers - layer - 1], hidden_dim[self.num_layers - layer - 1], output_dim, num_hidden = 2))
            reconstruct_degree.append(MLP(hidden_dim[self.num_layers - layer - 1], hidden_dim[self.num_layers - layer - 1], 1, num_hidden = 3))
        self.GNN = nn.Sequential(*GNNEncoder)
        self.decode_mean = nn.Sequential(*decode_mean)
        self.decode_std = nn.Sequential(*decode_std)
        self.reconstruct_self = nn.Sequential(*reconstruct_self)
        self.reconstruct_degree = nn.Sequential(*reconstruct_degree)
        self.degree_loss_function = nn.MSELoss()
        self.self_loss_function = nn.MSELoss()
        self.neighbor_loss_function = nn.MSELoss()

    def encoder(self, h0, adj):
        '''
        GNN encoder
        '''
        h_list = []
        h = h0
        # 经过多层聚合，获得最终的节点表示
        for layer in range(self.num_layers):
            h = self.GNN[layer]([h, adj])
            h_list.append(h)
        return h_list
    
    # 邻域分布计算方法
    def neighborhood_distribution(self, embedding, neighbor_dict, degree):
        mean = torch.zeros(embedding.shape, device = self.device)
        for node in neighbor_dict.keys():
            neighbors = neighbor_dict[node]
            embedding_neighbor = torch.reshape(embedding[node, :], [1, -1])
            embedding_neighbor = torch.cat([embedding_neighbor, embedding[neighbors, :]], dim = 0)
            mean[node, :] = torch.mean(embedding_neighbor, dim = 0)
        return mean
    
    def decoder(self, h_list, h0, degree, neighbor_dict):
        '''
         Inv-GNN decoder
        '''
        self.loss_self = 0. # 自重构损失
        self.kl_neighbor = 0. # 邻居重构KL散度
        self.loss_degree = 0. # 度数重构损失
        mean_prior = 0.
        z_list = []
        # 从最后一层开始，逐层解码
        for layer in range(self.num_layers - 1, -1, -1):
            if layer == 0:
                target_embedding = h0
            else:
                target_embedding = h_list[layer - 1]
            # reconstruct self node 自节点重构
            mean = self.reconstruct_self[self.num_layers - layer - 1](h_list[layer])
            if layer < self.num_layers - 1:
                mean_prior = self.decode_mean[self.num_layers - layer - 2](mean_prior)
            mean_posterior = mean + mean_prior # 后验均值
            # 计算潜在变量z
            log_std = self.decode_std[self.num_layers - layer - 1](h_list[layer]) # 对数标准差
            z = mean_posterior + torch.randn(mean.shape).to(self.device) * log_std.exp() # [q, c]
            # 自重构损失
            self.loss_self += self.self_loss_function(target_embedding, z)
            # reconstruct neighbors 邻居重构
            h_mean = self.neighborhood_distribution(target_embedding, neighbor_dict, degree)
            self.kl_neighbor += self.kl_normal(mean_posterior, log_std, h_mean)
            # reconstruct degree 度数重构
            reconstruction_degree = F.relu(self.reconstruct_degree[self.num_layers - layer - 1](h_list[layer])) # non-negative transformation
            self.loss_degree += self.degree_loss_function(reconstruction_degree, torch.unsqueeze(degree, dim = 1).float())
            
            mean_prior = z
            z_list.append(z)
        # 计算平均损失并加权求和得到总损失
        self.loss_self = self.loss_self / self.num_layers
        self.kl_neighbor = self.kl_neighbor / self.num_layers
        self.loss_degree = self.loss_degree / self.num_layers
        loss = self.lambda_self * self.loss_self + self.lambda_neighbor * self.kl_neighbor + self.lambda_degree * self.loss_degree
        return loss

    def forward(self, adj, h0, degree, neighbor_dict):
        # Generate GNN embeddings
        h_list = self.encoder(h0, adj)
        # Decoding and generating the latent representation by decoder
        loss = self.decoder(h_list, h0, degree, neighbor_dict)
        return loss, h_list[-1]
    
    # 计算两个正态分布之间的KL散度
    def kl_normal(self, mean_source, log_sigma_source, mean_target):
        kl = 0.5 * torch.mean(-1 - 2 * log_sigma_source + torch.square(mean_source - mean_target) + torch.exp(2 * log_sigma_source))
        return kl
