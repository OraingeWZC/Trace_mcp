import os
import torch
import dgl
from tqdm import tqdm
from loguru import logger
from torch_sparse import SparseTensor

from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.mymodel_eval import evaluate
from tracegnn.models.gtrace.models.mymodel import MyTraceAnomalyModel, construct_neighbor_dict


def train_epoch(config: ExpConfig, loader: dgl.dataloading.GraphDataLoader,
                model: MyTraceAnomalyModel, optimizer):

    total_loss, structure_loss, latency_loss = 0.0, 0.0, 0.0
    total_steps = 0

    t = tqdm(loader) if config.enable_tqdm else loader

    for step, graph_batch in enumerate(t):
        graph_batch = graph_batch.to(config.device)

        # 添加调试信息
        # print(f"Train Debug - graph_batch.num_nodes(): {graph_batch.num_nodes()}")
        # print(f"Train Debug - graph_batch batch size: {graph_batch.batch_size}")

        adj_sparse = graph_batch.adjacency_matrix()
        adj = SparseTensor(
            row=adj_sparse.coalesce().indices()[0],
            col=adj_sparse.coalesce().indices()[1],
            sparse_sizes=adj_sparse.shape
        ).to(config.device)
        degree = adj.sum(0).to(config.device)
        neighbor_dict = construct_neighbor_dict(adj)

        pred = model(graph_batch, adj, degree, neighbor_dict)
        loss = pred["loss_total"]
        total_steps += 1

        # Set loss info
        total_loss += loss.item()
        structure_loss += pred["loss_structure"].item()
        latency_loss += pred["loss_latency"].item()
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()

        # Clip global grad to avoid nan
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Show loss
        if config.enable_tqdm:
            t.set_description(f"[Train] L:{loss.item():.3f} S:{pred['loss_structure']:.3f} "
                              f"Lc:{pred['loss_latency']:.3f}")

    return total_loss / total_steps, structure_loss / total_steps, latency_loss / total_steps


def val_epoch(config: ExpConfig, loader: dgl.dataloading.GraphDataLoader,
              model: MyTraceAnomalyModel):
    # Calculate total loss per epoch
    total_loss = 0.0
    structure_loss = 0.0
    latency_loss = 0.0
    total_steps = 0

    t = tqdm(loader) if config.enable_tqdm else loader

    model.eval()

    with torch.no_grad():
        for step, graph_batch in enumerate(t):
            graph_batch = graph_batch.to(config.device)
            adj_sparse = graph_batch.adjacency_matrix()
            adj = SparseTensor(
                row=adj_sparse.coalesce().indices()[0],
                col=adj_sparse.coalesce().indices()[1],
                sparse_sizes=adj_sparse.shape
            ).to(config.device)
            degree = adj.sum(0).to(config.device)
            neighbor_dict = construct_neighbor_dict(adj)

            pred = model(graph_batch, adj, degree, neighbor_dict)
            loss = pred["loss_total"]

            # Set loss info
            total_loss += loss.item()
            structure_loss += pred["loss_structure"].item()
            latency_loss += pred["loss_latency"].item()
            total_steps += 1

            # Show loss
            if config.enable_tqdm:
                t.set_description(f"[Val] L:{loss.item():.3f} S:{pred['loss_structure']:.3f} "
                                  f"Lc:{pred['loss_latency']:.3f}")
    
    model.train()
    
    return total_loss / total_steps, structure_loss / total_steps, latency_loss / total_steps


def trainer(config: ExpConfig,
            train_loader: dgl.dataloading.GraphDataLoader,
            val_loader: dgl.dataloading.GraphDataLoader,
            test_loader: dgl.dataloading.GraphDataLoader = None):

    # Define model and optimizer
    model = MyTraceAnomalyModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters())

    # Set min loss for evaluation
    min_val_loss = 1e9

    # Train  model
    logger.info("Start training...")
    for epoch in range(config.max_epochs):
        logger.info(f'-------------> Epoch {epoch}')

        # Train 1 epoch
        logger.info(f'Training...')

        train_total, train_struct, train_lat = train_epoch(config, train_loader, model, optimizer)
        # Set output
        logger.info(f'Train Epoch: {epoch}  Loss: {train_total} Structure Loss: {train_struct} Latency Loss: {train_lat}')

        # DUBUG:取消Val，缩短时间
        # Val 1 epoch
        val_total, val_struct, val_lat = val_epoch(config, val_loader, model)
        # Set output
        logger.info(f'Valid Epoch: {epoch}  Loss: {val_total} Structure Loss: {val_struct} Latency Loss: {val_lat}')

        # If validation loss is smaller than min_val_loss, save the model
        
        # DEBUG:取消Val，缩短时间
        if val_total < min_val_loss:
            min_val_loss = val_total

            if test_loader is not None:
                logger.info("Valid loss is smaller. Start evaluation...")
                evaluate(config, test_loader, model)
            
            # Save model
            logger.info("Valid loss is smaller. Model saved.")
            torch.save(model.state_dict(), "model.pth")
        
    # Final evaluation
    if test_loader is not None:
        evaluate(config, test_loader, model)
        
    logger.info("Training finished.")
