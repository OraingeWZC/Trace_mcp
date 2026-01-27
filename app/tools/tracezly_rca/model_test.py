import warnings
warnings.filterwarnings("ignore")

import os
import torch
import dgl
import numpy as np
from loguru import logger

import mltk
from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.dataset import TestDataset, init_config
from tracegnn.models.gtrace.models.mymodel import MyTraceAnomalyModel
from tracegnn.models.gtrace.mymodel_eval import evaluate


@torch.no_grad()
def test_evaluate():
    """
    测试 mymodel_eval 中的 evaluate 函数
    """
    # 使用 Experiment 上下文管理器加载配置
    with mltk.Experiment(ExpConfig) as exp:
        config: ExpConfig = exp.config

        # 加载测试数据集
        logger.info('Loading test dataset...')
        test_dataset = TestDataset(config)
        
        # 初始化配置参数
        init_config(config)
        
        # 创建数据加载器
        test_loader = dgl.dataloading.GraphDataLoader(
            test_dataset, 
            batch_size=config.test_batch_size,  # 使用测试批大小
            shuffle=True  # 保持顺序以便分析
        )

        # 初始化模型
        logger.info('Initializing model...')
        model = MyTraceAnomalyModel(config).to(config.device)

        # 加载训练好的模型权重
        model_path = "model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # 加载模型状态字典
        logger.info(f'Loading model weights from {model_path}...')
        try:
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            logger.info('Model weights loaded successfully.')
        except RuntimeError as e:
            logger.error(f"Error loading model weights: {e}")
            logger.info("Trying to load with strict=False...")
            model.load_state_dict(torch.load(model_path, map_location=config.device), strict=False)
            logger.info('Model weights loaded with strict=False.')

        model.eval()
        logger.info(f'Model loaded and set to evaluation mode')

        # 调用 evaluate 函数进行评估和根因分析
        logger.info('Starting evaluation with root cause analysis...')
        try:
            evaluate(config, test_loader, model)
            logger.info('Evaluation completed successfully.')
        except Exception as e:
            logger.error(f'Error during evaluation: {e}')
            import traceback
            traceback.print_exc()


def test_evaluate_with_subset():
    """
    测试 evaluate 函数，但只使用数据集的一个小子集以节省时间
    """
    # 使用 Experiment 上下文管理器加载配置
    with mltk.Experiment(ExpConfig) as exp:
        config: ExpConfig = exp.config

        # 加载测试数据集
        logger.info('Loading test dataset...')
        test_dataset = TestDataset(config)
        
        # 初始化配置参数
        init_config(config)
        
        # 为了节省时间，选择包含异常样本的子集
        normal_indices = []
        anomaly_indices = []
        
        # 遍历数据集，找到正常和异常样本
        logger.info('Searching for normal and anomaly samples...')
        max_samples = len(test_dataset)  # 限制搜索范围以节省时间
        for i in range(max_samples):
            try:
                # 直接从数据库获取图数据以检查是否为异常样本
                graph_data = test_dataset.test_db.get(i)
                if graph_data.anomaly == 0:
                    normal_indices.append(i)
                else:
                    anomaly_indices.append(i)
            except Exception as e:
                logger.warning(f'Error loading sample {i}: {e}')
                continue
        
        # 选择最多5个正常样本和5个异常样本
        selected_normal = normal_indices[:5]
        selected_anomaly = anomaly_indices[:5]
        subset_indices = selected_normal + selected_anomaly
        
        logger.info(f'Found {len(selected_normal)} normal samples and {len(selected_anomaly)} anomaly samples')
        logger.info(f'Using subset of {len(subset_indices)} samples for quick testing: {subset_indices}')
        
        # 创建子集数据集
        class SubsetDataset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        subset_dataset = SubsetDataset(test_dataset, subset_indices)
        
        # 创建数据加载器
        test_loader = dgl.dataloading.GraphDataLoader(
            subset_dataset, 
            batch_size=config.test_batch_size,
            shuffle=False
        )

        # 初始化模型
        logger.info('Initializing model...')
        model = MyTraceAnomalyModel(config).to(config.device)

        # 加载训练好的模型权重
        model_path = "model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # 加载模型状态字典
        logger.info(f'Loading model weights from {model_path}...')
        try:
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            logger.info('Model weights loaded successfully.')
        except RuntimeError as e:
            logger.error(f"Error loading model weights: {e}")
            logger.info("Trying to load with strict=False...")
            model.load_state_dict(torch.load(model_path, map_location=config.device), strict=False)
            logger.info('Model weights loaded with strict=False.')

        model.eval()
        logger.info(f'Model loaded and set to evaluation mode')

        # 调用 evaluate 函数进行评估和根因分析
        logger.info('Starting evaluation with root cause analysis (subset) ...')
        try:
            evaluate(config, test_loader, model)
            logger.info('Evaluation completed successfully.')
        except Exception as e:
            logger.error(f'Error during evaluation: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    print("Choose test mode:")
    print("1. Full evaluation (may take longer)")
    print("2. Subset evaluation (faster)")
    
    choice = input("Enter choice (1 or 2, default 2): ").strip()
    
    if choice == "1":
        test_evaluate()
    else:
        test_evaluate_with_subset()