import warnings
warnings.filterwarnings("ignore")

import os
import torch
import dgl
import numpy as np
from loguru import logger

# 设置日志级别为DEBUG以便看到详细的调试信息
logger.remove()
logger.add(lambda msg: print(msg, end=''), level="DEBUG")
from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.dataset import init_config
from tracegnn.models.gtrace.models.mymodel import MyTraceAnomalyModel, construct_neighbor_dict
from tracegnn.data.trace_graph import TraceGraph
from tracegnn.data.graph_to_vector import graph_to_dgl
from tracegnn.utils.analyze_root_cause import evaluate_with_root_cause, load_service_id_to_name
import mltk
from torch_sparse import SparseTensor

@torch.no_grad()
def main():
    # 使用 Experiment 上下文管理器加载配置
    with mltk.Experiment(ExpConfig) as exp:
        config: ExpConfig = exp.config
        
        # 确保使用CPU以避免潜在的GPU内存问题
        config.device = 'cpu'
        
        # 初始化配置参数
        logger.info('Initializing config parameters...')
        init_config(config)
        
        # 打印当前DatasetParams
        logger.info(f"Dataset parameters: operation_cnt={config.DatasetParams.operation_cnt}, "
                   f"service_cnt={config.DatasetParams.service_cnt}, "
                   f"status_cnt={config.DatasetParams.status_cnt}")

        # 初始化模型
        logger.info('Initializing model...')
        model = MyTraceAnomalyModel(config).to(config.device)

        # 加载训练好的模型权重
        model_path = "model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info(f'Loading model from {model_path}...')
        
        # 加载模型权重字典
        checkpoint = torch.load(model_path, map_location=config.device)
        
        # 获取当前模型的参数名
        model_state_dict = model.state_dict()
        
        # 创建一个新的参数字典，只保留形状匹配的参数
        filtered_checkpoint = {}
        skipped_params = 0
        for name, param in checkpoint.items():
            if name in model_state_dict:
                # 检查形状是否匹配
                if param.shape == model_state_dict[name].shape:
                    filtered_checkpoint[name] = param
                else:
                    # 对于嵌入层权重，如果维度不同，我们尝试截断或填充
                    if 'embedding' in name and len(param.shape) == 2 and len(model_state_dict[name].shape) == 2:
                        # 确保第二个维度相同
                        if param.shape[1] == model_state_dict[name].shape[1]:
                            # 截断或填充第一个维度
                            if param.shape[0] > model_state_dict[name].shape[0]:
                                # 截断
                                filtered_checkpoint[name] = param[:model_state_dict[name].shape[0]]
                                logger.warning(f'Truncated parameter {name}: '\
                                             f'from {param.shape} to {model_state_dict[name].shape}')
                            else:
                                # 填充
                                padded_param = torch.zeros_like(model_state_dict[name])
                                padded_param[:param.shape[0]] = param
                                filtered_checkpoint[name] = padded_param
                                logger.warning(f'Padded parameter {name}: '\
                                             f'from {param.shape} to {model_state_dict[name].shape}')
                        else:
                            skipped_params += 1
                            logger.warning(f'Skipping parameter {name} due to shape mismatch: '\
                                          f'expected {model_state_dict[name].shape} vs actual {param.shape}')
                    else:
                        skipped_params += 1
                        logger.warning(f'Skipping parameter {name} due to shape mismatch: '\
                                      f'expected {model_state_dict[name].shape} vs actual {param.shape}')
            else:
                skipped_params += 1
                logger.warning(f'Skipping parameter {name} which is not in the current model')
        
        # 加载过滤后的参数
        model.load_state_dict(filtered_checkpoint, strict=False)
        model.eval()
        logger.info(f'Model loaded successfully (ignored {skipped_params} parameters)')

        # 直接实现简化版的评估逻辑，不依赖dataloader的复杂处理
        logger.info('Starting simplified evaluation...')
        
        # 加载测试数据集
        processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
        test_path = os.path.join(processed_dir, config.test_dataset)

        # 加载 service_id 到 service_name 的映射
        service_id_yaml_path = os.path.join(processed_dir, 'service_id.yml')
        try:
            service_id_to_name = load_service_id_to_name(service_id_yaml_path)
            logger.info(f'Loaded service_id mapping from {service_id_yaml_path}')
        except Exception as e:
            logger.warning(f'Failed to load service_id mapping: {e}. Will use fallback.')
            service_id_to_name = None
        
        # 导入需要的类
        from tracegnn.data.bytes_db import BytesSqliteDB
        from tracegnn.data.trace_graph_db import TraceGraphDB
        
        # 打开数据库
        with TraceGraphDB(BytesSqliteDB(test_path)) as db:
            # 只评估前10个样本
            num_samples = len(db)
            logger.info(f'Evaluating on {num_samples} test samples...')

            # 用于存储结果
            total_losses = []
            structure_losses = []
            latency_losses = []
            labels = []

            # 用于根因定位
            all_trace_info = []
            true_root_causes_dict = {}

            # 处理每个样本
            for i in range(num_samples):
                try:
                    # 获取图数据
                    graph: TraceGraph = db.get(i)
                    
                    # 转换为DGL图
                    dgl_graph = graph_to_dgl(graph)
                    dgl_graph = dgl_graph.to(config.device)
                    
                    # 构建邻接矩阵和其他需要的参数
                    adj_sparse = dgl_graph.adjacency_matrix()
                    adj = SparseTensor(
                        row=adj_sparse.coalesce().indices()[0],
                        col=adj_sparse.coalesce().indices()[1],
                        sparse_sizes=adj_sparse.shape
                    ).to(config.device)
                    degree = adj.sum(0).to(config.device)
                    neighbor_dict = construct_neighbor_dict(adj)
                    
                    # 模型前向传播
                    pred = model(dgl_graph, adj, degree, neighbor_dict, n_z=config.Model.n_z)

                    # 收集结果
                    total_losses.append(pred['loss_total'].item())
                    structure_losses.append(pred['loss_structure'].item())
                    latency_losses.append(pred['loss_latency'].item())
                    labels.append(graph.anomaly)

                    # 计算节点级分数
                    if 'node_structure_scores' in pred and 'node_latency_scores' in pred:
                        combined_node_scores = (
                            pred['alpha'] * pred['node_structure_scores'] +
                            pred['beta'] * pred['node_latency_scores']
                        )
                        trace_node_count = dgl_graph.num_nodes()
                        normalized_node_scores = combined_node_scores / trace_node_count
                    else:
                        normalized_node_scores = torch.zeros(dgl_graph.num_nodes(), device=config.device)

                    # 生成trace_id (使用简单的索引作为ID)
                    trace_id = f"trace_{i}"

                    # 收集trace信息用于根因定位
                    trace_info = {
                        'graph': dgl_graph,
                        'trace_id': trace_id,
                        'is_anomalous': graph.anomaly > 0,
                        'node_scores': normalized_node_scores,
                    }
                    all_trace_info.append(trace_info)

                    # 收集真实根因（如果存在）
                    if graph.anomaly > 0 and hasattr(graph, 'root_cause') and graph.root_cause is not None:
                        true_root_causes_dict[trace_id] = graph.root_cause
                        # logger.debug(f'Sample {i} - Root cause: {graph.root_cause}')

                    # logger.info(f'Sample {i+1}/{num_samples}: loss_total={pred["loss_total"].item():.4f}, '\
                    #             f'loss_structure={pred["loss_structure"].item():.4f}, '\
                    #             f'loss_latency={pred["loss_latency"].item():.4f}, '\
                    #             f'anomaly={graph.anomaly}')

                except Exception as e:
                    logger.error(f'Error processing sample {i}: {str(e)}')
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # 计算简单统计
            if total_losses:
                logger.info('\n' + '='*60)
                logger.info('Evaluation Summary:')
                logger.info(f'Average total loss: {np.mean(total_losses):.4f}')
                logger.info(f'Average structure loss: {np.mean(structure_losses):.4f}')
                logger.info(f'Average latency loss: {np.mean(latency_losses):.4f}')
                logger.info(f'Total anomalous samples: {sum(1 for l in labels if l > 0)} out of {len(labels)}')
            else:
                logger.warning('No samples were successfully evaluated.')

            # 根因定位评估
            if all_trace_info:
                logger.info('\n' + '='*60)
                logger.info('Root Cause Analysis')
                logger.info('='*60)

                # 打印数据采集情况
                anomalous_count = sum(1 for t in all_trace_info if t['is_anomalous'])
                logger.info(f'Total traces: {len(all_trace_info)}')
                logger.info(f'Anomalous traces: {anomalous_count}')
                logger.info(f'Traces with ground truth root causes: {len(true_root_causes_dict)}')

                # 打印一些样例信息
                if len(all_trace_info) > 0:
                    sample_trace = all_trace_info[0]
                    logger.info(f'\nSample trace info:')
                    logger.info(f'  - trace_id: {sample_trace["trace_id"]}')
                    logger.info(f'  - is_anomalous: {sample_trace["is_anomalous"]}')
                    logger.info(f'  - node_scores shape: {sample_trace["node_scores"].shape}')
                    logger.info(f'  - graph has service_id: {"service_id" in sample_trace["graph"].ndata}')

                    if 'service_id' in sample_trace['graph'].ndata:
                        service_ids = sample_trace['graph'].ndata['service_id'].cpu().numpy()
                        logger.info(f'  - service_ids sample: {service_ids[:5]}')
                        logger.info(f'  - unique service count: {len(np.unique(service_ids))}')

                if len(true_root_causes_dict) > 0:
                    sample_key = list(true_root_causes_dict.keys())[0]
                    sample_value = true_root_causes_dict[sample_key]
                    logger.info(f'\nSample ground truth:')
                    logger.info(f'  - trace_id: {sample_key}')
                    logger.info(f'  - root_cause type: {type(sample_value)}')
                    logger.info(f'  - root_cause value: {sample_value}')

                # 执行根因定位评估
                try:
                    _, acc_top1, acc_top3, acc_top5 = evaluate_with_root_cause(
                        all_trace_info,
                        true_root_causes=true_root_causes_dict,
                        topk=5,
                        service_id_to_name=service_id_to_name
                    )
                    logger.info(f'\n' + '='*60)
                    logger.info(f'Root Cause Localization Results:')
                    logger.info(f'  - Top-1 Accuracy: {acc_top1:.4f}')
                    logger.info(f'  - Top-3 Accuracy: {acc_top3:.4f}')
                    logger.info(f'  - Top-5 Accuracy: {acc_top5:.4f}')
                    logger.info('='*60)
                except Exception as e:
                    logger.error(f'Error during root cause analysis: {str(e)}')
                    import traceback
                    logger.error(traceback.format_exc())

        logger.info('\nEvaluation completed successfully.')

if __name__ == '__main__':
    main()