import os
import torch
import dgl
import numpy as np
from loguru import logger
from torch_sparse import SparseTensor

from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.dataset import TestDataset, init_config
from tracegnn.models.gtrace.models.mymodel import MyTraceAnomalyModel, construct_neighbor_dict
from tracegnn.utils.analyze_nll import analyze_anomaly_scores
import mltk


@torch.no_grad()
def test_and_save_scores():
    # 使用 Experiment 上下文管理器加载配置
    with mltk.Experiment(ExpConfig) as exp:
        config: ExpConfig = exp.config

        # 加载测试数据集
        logger.info('Loading test dataset...')
        test_dataset = TestDataset(config)
        
        # 初始化配置参数
        init_config(config)
        
        test_loader = dgl.dataloading.GraphDataLoader(
            test_dataset, batch_size=config.batch_size)

        # 初始化模型
        logger.info('Initializing model...')
        model = MyTraceAnomalyModel(config).to(config.device)

        # 加载训练好的模型权重
        model_path = "model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        logger.info(f'Model loaded from {model_path}')

        # 存储分数和标签
        total_scores = []
        structure_scores = []
        latency_scores = []
        graph_labels = []

        logger.info('Starting evaluation...')

        # 遍历测试集
        for label_graphs, test_graphs, graph_label_tensors in test_loader:
            # 将数据移到设备上
            test_graphs = test_graphs.to(config.device)
            label_graphs = label_graphs.to(config.device)
            graph_label_tensors = graph_label_tensors.to(config.device)

            # 构建邻接矩阵
            adj_sparse = test_graphs.adjacency_matrix()
            adj = SparseTensor(
                row=adj_sparse.coalesce().indices()[0],
                col=adj_sparse.coalesce().indices()[1],
                sparse_sizes=adj_sparse.shape
            ).to(config.device)
            degree = adj.sum(0).to(config.device)
            neighbor_dict = construct_neighbor_dict(adj)

            # 模型前向传播
            preds = model(test_graphs, adj, degree, neighbor_dict)

            # 处理每个图的预测结果
            for i in range(min(5, test_graphs.batch_size)):  # 修复索引越界问题
                # 获取图标签 (0=正常, 1=结构异常, 2=延迟异常)
                if graph_label_tensors[i, 0]:  # 结构异常
                    graph_label = 1
                elif graph_label_tensors[i, 1]:  # 延迟异常
                    graph_label = 2
                else:  # 正常
                    graph_label = 0

                graph_labels.append(graph_label)

                # 收集分数
                total_scores.append(preds['loss_total'].item())
                structure_scores.append(preds['loss_structure'].item())
                latency_scores.append(preds['loss_latency'].item())

        # 转换为numpy数组
        total_scores = np.array(total_scores, dtype=np.float32)
        structure_scores = np.array(structure_scores, dtype=np.float32)
        latency_scores = np.array(latency_scores, dtype=np.float32)
        graph_labels = np.array(graph_labels, dtype=np.int32)

        # 保存分数和标签到文件
        logger.info('Saving scores and labels to files...')
        np.savez('test_scores.npz',
                 total_scores=total_scores,
                 structure_scores=structure_scores,
                 latency_scores=latency_scores,
                 graph_labels=graph_labels)

        # 分析总体分数
        logger.info('Analyzing scores...')
        result = analyze_anomaly_scores(
            score_list=total_scores,
            label_list=graph_labels,
            method='gtrace',
            dataset=config.dataset,
            save_dict=True,
            save_filename='test_results.csv'
        )

        # 输出最佳F1分数和对应的阈值
        score_normal = result['score_normal']
        score_structure = result['score_structure']
        score_latency = result['score_latency']
        best_fscore = result['best_fscore']
        best_threshold = result['best_threshold']
        median = result['median']
        mad = result['mad']
        fscore = result['fscore']
        threshold = result['threshold']
        logger.info(f'Score Normal: {score_normal:.4f}')
        logger.info(f'Score Structure: {score_structure:.4f}')
        logger.info(f'Score Latency: {score_latency:.4f}')
        logger.info(f'Best F1 Score: {best_fscore:.4f}')
        logger.info(f'Best Threshold: {best_threshold:.4f}')
        logger.info(f'median: {median:.4f}')
        logger.info(f'mad: {mad:.4f}')
        logger.info(f'Fscore: {fscore:.4f}')
        logger.info(f'Threshold: {threshold:.4f}')


        logger.info('Results saved:')
        logger.info('- test_scores.npz: Anomaly scores and labels')
        logger.info('- test_results.csv: Detailed evaluation results')


if __name__ == '__main__':
    test_and_save_scores()