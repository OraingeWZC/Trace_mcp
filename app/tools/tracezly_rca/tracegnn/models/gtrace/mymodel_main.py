import warnings
warnings.filterwarnings("ignore")

import random
import os
import numpy as np
from typing import *

import mltk
import dgl
import torch
import torch.backends.cudnn
from loguru import logger

from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.dataset import TrainDataset, TestDataset
from tracegnn.models.gtrace.mymodel_train import trainer


def init_seed(config: ExpConfig):
    # set random seed to encourage reproducibility (does it really work?)
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(config.seed)
        random.seed(config.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main(exp: mltk.Experiment):
    config: ExpConfig = exp.config

    # init
    init_seed(config)
    logger.info(f"Device: {config.device}")

    # Load dataset
    logger.info(f"Loading dataset {config.dataset} ({config.test_dataset})...")
    # DEBUG: 这里为了快速验证，把train_dataset的valid参数设置为True，实际应该是false
    train_dataset = TrainDataset(config, valid=False)
    val_dataset = TrainDataset(config, valid=True)

    # Check if the test path exists
    test_flag = os.path.join(config.dataset_root_dir, config.dataset, "processed", config.test_dataset)
    if test_flag:
        test_dataset = TestDataset(config)
    else:
        test_dataset = None
    
    train_loader = dgl.dataloading.GraphDataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = dgl.dataloading.GraphDataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=True)
    
    if test_dataset is not None:
        test_loader = dgl.dataloading.GraphDataLoader(
            test_dataset, batch_size=config.batch_size)
    else:
        test_loader = None
    
    # Train
    trainer(config=exp.config, 
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader)
    

if __name__ == '__main__':
    with mltk.Experiment(ExpConfig) as exp:
        main(exp)

# DEBUG
# with mltk.Experiment(ExpConfig) as exp:
#     main(exp)