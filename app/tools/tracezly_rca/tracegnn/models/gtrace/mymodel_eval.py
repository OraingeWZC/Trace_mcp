from typing import *

from tracegnn.models.gtrace.models.mymodel import MyTraceAnomalyModel, construct_neighbor_dict
from .config import ExpConfig
from tracegnn.utils.analyze_root_cause import evaluate_with_root_cause, load_service_id_to_name

import dgl
from loguru import logger
import torch
import numpy as np
from torch_sparse import SparseTensor
from .utils import dgl_graph_key
from tracegnn.utils.analyze_nll import analyze_anomaly_scores



@torch.no_grad()
def evaluate(config: ExpConfig, dataloader: dgl.dataloading.GraphDataLoader, model: MyTraceAnomalyModel):
    """
    Evaluate MyTraceAnomalyModel (基于 total_loss / structure_loss / latency_loss)
    同时进行根因定位分析
    """
    device = config.device
    n_z = config.Model.n_z

    # 加载 service_id 到 service_name 的映射
    import os
    processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
    service_id_yaml_path = os.path.join(processed_dir, 'service_id.yml')
    try:
        service_id_to_name = load_service_id_to_name(service_id_yaml_path)
        logger.info(f'Loaded service_id mapping for root cause analysis')
    except Exception as e:
        logger.warning(f'Failed to load service_id mapping: {e}. Will use fallback.')
        service_id_to_name = None

    # Train model
    logger.info('Start Evaluation with nll...')
    model.eval()

    anomaly_score_list = []
    graph_label_list = []
    all_trace_info = []
    true_root_causes_dict = {}

    with torch.no_grad():
        t = dataloader
        if config.enable_tqdm:
            from tqdm import tqdm
            t = tqdm(dataloader)

        for batch_idx, batch in enumerate(t):
            # batch: (graphs, labels)
            if isinstance(batch, (tuple, list)):
                test_graphs, graph_anomaly_labels, root_causes, fault_categories = batch
            else:
                test_graphs = batch
                graph_anomaly_labels = batch.ndata['label'] if 'label' in batch.ndata else None

            # Empty cache first
            if 'cuda' in config.device:
                torch.cuda.empty_cache()

            test_graphs = test_graphs.to(device)
            if graph_anomaly_labels is not None:
                graph_anomaly_labels = graph_anomaly_labels.to(device)

            test_graph_list: List[dgl.DGLGraph] = dgl.unbatch(test_graphs)

            for i, single_test_graph in enumerate(test_graph_list):
                graph_key = dgl_graph_key(single_test_graph)
                single_graph_anomaly_label = (
                    graph_anomaly_labels[i].item() if graph_anomaly_labels is not None else
                    (single_test_graph.graph_label if hasattr(single_test_graph, 'graph_label') else 0)
                )

                # 获取真实根因
                groundtruth_root_cause = None
                if isinstance(batch, (tuple, list)) and len(batch) > 2:
                    # 从数据加载器的批次数据中获取root_cause
                    groundtruth_root_cause = root_causes[i].item() if hasattr(root_causes[i], 'item') else root_causes[i]
                elif hasattr(single_test_graph, 'root_cause'):
                    # 从图对象属性中获取root_cause
                    groundtruth_root_cause = single_test_graph.root_cause

                # 只有在是异常trace时才添加到true_root_causes_dict中
                if int(single_graph_anomaly_label) > 0:
                    true_root_causes_dict[graph_key] = groundtruth_root_cause

                # 构造邻接矩阵
                adj_sparse = single_test_graph.adjacency_matrix()
                adj = SparseTensor(
                    row=adj_sparse.coalesce().indices()[0],
                    col=adj_sparse.coalesce().indices()[1],
                    sparse_sizes=adj_sparse.shape
                ).to(device)
                degree = adj.sum(0).to(device)
                neighbor_dict = construct_neighbor_dict(adj)

                # 运行模型
                pred = model(single_test_graph, adj, degree, neighbor_dict, n_z=n_z)

                # 计算nll
                # 结构NLL就是loss_structure
                nll_structure = pred['loss_structure']
                # 延迟NLL就是loss_latency
                nll_latency = pred['loss_latency']

                # NEW
                # # Combine structure and latency NLL
                # combined_nll = nll_structure.item() + nll_latency.item()
                weighted_structure = (pred['alpha'] * nll_structure).item()
                weighted_latency = (pred['beta'] * nll_latency).item()
                anomaly_score = weighted_structure + weighted_latency

                anomaly_score_list.append(anomaly_score)
                graph_label_list.append(int(single_graph_anomaly_label))

                # 节点级分数
                if 'node_structure_scores' in pred and 'node_latency_scores' in pred:
                    # NEW: 应用"Reducing the Entropy Gap"优化原则：组合结构和延迟得分
                    combined_node_scores = (
                        pred['alpha'] * pred['node_structure_scores'] +
                        pred['beta'] * pred['node_latency_scores']
                    )
                    
                    # 应用"Reducing the Entropy Gap"优化原则：节点数量归一化
                    # 将每个节点得分除以trace的总节点数
                    trace_node_count = single_test_graph.num_nodes()
                    normalized_node_scores = combined_node_scores / trace_node_count
                    
                    single_test_graph.ndata['node_anomaly_score'] = normalized_node_scores
                    single_test_graph.ndata['node_structure_score'] = pred['node_structure_scores']
                    single_test_graph.ndata['node_latency_score'] = pred['node_latency_scores']
                    current_node_scores = normalized_node_scores
                else:
                    logger.warning(f"Graph {graph_key}: No node scores in prediction output.")
                    current_node_scores = torch.zeros(single_test_graph.num_nodes(), device=device)

                # 收集trace信息
                trace_info = {
                    'graph': single_test_graph,
                    'trace_id': graph_key,
                    'is_anomalous': int(single_graph_anomaly_label) > 0,
                    'node_scores': current_node_scores,
                }
                all_trace_info.append(trace_info)

        # Convert to numpy arrays
        anomaly_score_array = np.array(anomaly_score_list, dtype=np.float32)
        graph_label_array = np.array(graph_label_list, dtype=np.int64)

        # Debug information
        # logger.debug(f'Combined NLL range: [{combined_nll_array.min():.2f}, {combined_nll_array.max():.2f}]')
        # logger.debug(f'Graph labels distribution: {np.bincount(graph_label_array)}')

        # Check for any abnormally large NLL values
        # normal_nll_values = anomaly_score_array[graph_label_array == 0]
        # if len(normal_nll_values) > 0:
        #     logger.debug(f'Normal NLL range: [{normal_nll_values.min():.2f}, {normal_nll_values.max():.2f}]')
        #     logger.debug(f'Normal NLL mean: {np.mean(normal_nll_values):.2f}')

        # Set evaluation output
        logger.info('-------------------Graph Level Overall-----------------------')
        # Get overall graph level result
        overall_result = analyze_anomaly_scores(
            score_list=anomaly_score_array,
            label_list=graph_label_array
        )
        logger.info(overall_result)

        # 根因定位评估
        if all_trace_info:
            logger.info("-------------------Root Cause Analysis-----------------------")
            root_cause_results, acc_top1, acc_top3, acc_top5 = evaluate_with_root_cause(
                all_trace_info,
                true_root_causes=true_root_causes_dict,
                topk=5,
                service_id_to_name=service_id_to_name
            )
            logger.info(f"Top1根因定位准确率: {acc_top1:.4f}，Top3根因定位准确率: {acc_top3:.4f}，Top5根因定位准确率: {acc_top5:.4f}")

    model.train()