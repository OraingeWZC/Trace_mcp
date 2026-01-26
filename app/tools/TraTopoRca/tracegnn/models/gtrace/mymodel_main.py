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

    # Check if the test path exists.
    # NOTE: Previously this was a string truthiness check (always True), causing it to always try loading TestDataset.
    test_path = os.path.join(config.dataset_root_dir, config.dataset, "processed", config.test_dataset)
    if os.path.exists(test_path):
        test_dataset = TestDataset(config)
    else:
        logger.warning(f"Test dataset path not found, skip evaluation loader: {test_path}")
        test_dataset = None

    # Optional debug: print RuntimeInfo latency stats once after init_config() is populated by TrainDataset.
    try:
        if bool(getattr(config, "debug_nan", False)) and getattr(config, "RuntimeInfo", None) is not None:
            lr = getattr(config.RuntimeInfo, "latency_range", None)
            p98 = getattr(config.RuntimeInfo, "latency_p98", None)
            if lr is not None:
                logger.info(
                    "[Debug] latency_range: shape=%s finite=%s min=%.6f max=%.6f",
                    tuple(lr.shape),
                    bool(torch.isfinite(lr).all().item()),
                    float(lr.min().item()),
                    float(lr.max().item()),
                )
                std = lr[:, 1]
                frac_std10 = float((std == 10.0).float().mean().item())
                logger.info(
                    "[Debug] latency_range std: finite=%s min=%.6f max=%.6f frac(std==10)=%.4f",
                    bool(torch.isfinite(std).all().item()),
                    float(std.min().item()),
                    float(std.max().item()),
                    frac_std10,
                )
            if p98 is not None:
                logger.info(
                    "[Debug] latency_p98: shape=%s finite=%s min=%.6f max=%.6f",
                    tuple(p98.shape),
                    bool(torch.isfinite(p98).all().item()),
                    float(p98.min().item()),
                    float(p98.max().item()),
                )
    except Exception as e:
        logger.warning(f"[Debug] Skip printing latency stats: {e}")
    
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
