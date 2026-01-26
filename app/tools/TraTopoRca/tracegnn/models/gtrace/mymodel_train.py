import os
import torch
import dgl
from tqdm import tqdm
from loguru import logger
from torch_sparse import SparseTensor
import os
import csv

from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.mymodel_eval import evaluate
from tracegnn.models.gtrace.models.mymodel import MyTraceAnomalyModel, construct_neighbor_dict


def train_epoch(config: ExpConfig, loader: dgl.dataloading.GraphDataLoader,
                model: MyTraceAnomalyModel, optimizer):

    total_loss, structure_loss, latency_loss = 0.0, 0.0, 0.0
    host_loss = 0.0
    total_steps = 0

    t = tqdm(loader) if config.enable_tqdm else loader

    for step, graph_batch in enumerate(t):
        # 保留设备端与CPU端各一份，用CPU端提取边，避免GPU边导出触发非法访问
        graph_dev = graph_batch.to(config.device)
        graph_cpu = graph_batch if 'cuda' not in str(config.device) else graph_batch.to('cpu')

        # 添加调试信息
        # print(f"Train Debug - graph_batch.num_nodes(): {graph_batch.num_nodes()}")
        # print(f"Train Debug - graph_batch batch size: {graph_batch.batch_size}")

        # Build edge_index to avoid torch_sparse spmm on GPU; use scatter path in PyG
        u, v = graph_cpu.edges()
        # 构造给 GIN 的 edge_index（放在目标设备）
        u_dev = u.to(config.device).long().contiguous()
        v_dev = v.to(config.device).long().contiguous()
        edge_index = torch.stack([u_dev, v_dev], dim=0)
        N = int(graph_batch.num_nodes())
        # 在 CPU 端计算 degree，避免 GPU 内核在极端情况下非法访存
        v_cpu = v.long().contiguous().cpu()
        if v_cpu.numel() == 0:
            degree = torch.zeros(N, dtype=torch.long, device=config.device)
        else:
            vmax = int(v_cpu.max().item())
            if vmax >= N:
                deg_full = torch.bincount(v_cpu, minlength=vmax + 1)
                degree = torch.zeros(N, dtype=deg_full.dtype, device=config.device)
                degree[:min(N, deg_full.numel())] = deg_full[:min(N, deg_full.numel())]
            else:
                degree = torch.bincount(v_cpu, minlength=N).to(config.device)
        # 邻居字典用 CPU 索引构建
        neighbor_dict = construct_neighbor_dict((torch.stack([v_cpu*0 + 0, v_cpu], dim=0) if False else (edge_index.cpu(), N)))

        pred = model(graph_dev, edge_index, degree, neighbor_dict)
        loss = pred["loss_total"]
        total_steps += 1

        # Debug / safety: stop early on NaN/Inf to locate the first bad step.
        if bool(getattr(config, "debug_nan", False)):
            try:
                if not bool(torch.isfinite(loss).item()):
                    logger.error("[Debug] Non-finite loss_total at step=%s: %s", step, str(loss))
                    for k in ["loss_structure", "loss_latency", "loss_host"]:
                        v = pred.get(k, None) if isinstance(pred, dict) else None
                        if v is not None:
                            try:
                                logger.error("[Debug] %s=%s finite=%s", k, str(v), bool(torch.isfinite(v).all().item()))
                            except Exception:
                                logger.error("[Debug] %s=%s", k, str(v))
                    lv = pred.get("latency_logvar", None) if isinstance(pred, dict) else None
                    if lv is not None:
                        lv_det = lv.detach()
                        logger.error(
                            "[Debug] latency_logvar stats finite=%s min=%.6f max=%.6f",
                            bool(torch.isfinite(lv_det).all().item()),
                            float(lv_det.min().item()),
                            float(lv_det.max().item()),
                        )
                    raise RuntimeError("Non-finite loss detected (debug_nan=True).")
            except Exception:
                raise

        # Set loss info
        total_loss += loss.item()
        structure_loss += pred["loss_structure"].item()
        latency_loss += pred["loss_latency"].item()
        try:
            host_loss += float(pred.get('loss_host', 0.0))
        except Exception:
            pass
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()

        # Clip global grad to avoid nan
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Show loss
        if config.enable_tqdm:
            t.set_description(f"[Train] L:{loss.item():.3f} S:{pred['loss_structure']:.3f} "
                              f"Lc:{pred['loss_latency']:.3f} "
                              f"H:{float(pred.get('loss_host', 0.0)):.3f}")

    return total_loss / total_steps, structure_loss / total_steps, latency_loss / total_steps


def val_epoch(config: ExpConfig, loader: dgl.dataloading.GraphDataLoader,
              model: MyTraceAnomalyModel):
    # Calculate total loss per epoch
    total_loss = 0.0
    structure_loss = 0.0
    latency_loss = 0.0
    host_loss = 0.0
    total_steps = 0

    t = tqdm(loader) if config.enable_tqdm else loader

    model.eval()

    with torch.no_grad():
        for step, graph_batch in enumerate(t):
            graph_dev = graph_batch.to(config.device)
            graph_cpu = graph_batch if 'cuda' not in str(config.device) else graph_batch.to('cpu')
            # Build edge_index to avoid torch_sparse spmm on GPU; use scatter path in PyG
            u, v = graph_cpu.edges()
            u_dev = u.to(config.device).long().contiguous()
            v_dev = v.to(config.device).long().contiguous()
            edge_index = torch.stack([u_dev, v_dev], dim=0)
            N = int(graph_batch.num_nodes())
            v_cpu = v.long().contiguous().cpu()
            if v_cpu.numel() == 0:
                degree = torch.zeros(N, dtype=torch.long, device=config.device)
            else:
                vmax = int(v_cpu.max().item())
                if vmax >= N:
                    deg_full = torch.bincount(v_cpu, minlength=vmax + 1)
                    degree = torch.zeros(N, dtype=deg_full.dtype, device=config.device)
                    degree[:min(N, deg_full.numel())] = deg_full[:min(N, deg_full.numel())]
                else:
                    degree = torch.bincount(v_cpu, minlength=N).to(config.device)
            neighbor_dict = construct_neighbor_dict((edge_index.cpu(), N))

            pred = model(graph_dev, edge_index, degree, neighbor_dict)
            loss = pred["loss_total"]

            # Set loss info
            total_loss += loss.item()
            structure_loss += pred["loss_structure"].item()
            latency_loss += pred["loss_latency"].item()
            total_steps += 1
            try:
                host_loss += float(pred.get('loss_host', 0.0))
            except Exception:
                pass

            # Show loss
            if config.enable_tqdm:
                t.set_description(f"[Val] L:{loss.item():.3f} S:{pred['loss_structure']:.3f} "
                                  f"Lc:{pred['loss_latency']:.3f} "
                                  f"H:{float(pred.get('loss_host', 0.0)):.3f}")
    
    model.train()
    
    return total_loss / total_steps, structure_loss / total_steps, latency_loss / total_steps


def trainer(config: ExpConfig,
            train_loader: dgl.dataloading.GraphDataLoader,
            val_loader: dgl.dataloading.GraphDataLoader,
            test_loader: dgl.dataloading.GraphDataLoader = None):

    # Define model and optimizer
    model = MyTraceAnomalyModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters())

    # Freeze Kendall log_sigma in early epochs to avoid weight drift
    freeze_epochs = int(getattr(getattr(config, 'Model'), 'freeze_sigma_epochs', 0)) if hasattr(config, 'Model') else 0
    if freeze_epochs > 0:
        for p in [getattr(model, 'log_sigma_structure', None), getattr(model, 'log_sigma_latency', None), getattr(model, 'log_sigma_host', None)]:
            if p is not None:
                p.requires_grad = False

    # Set min loss for evaluation
    min_val_loss = 1e9

    # Train  model
    logger.info("Start training...")

    # Prepare training dynamics log path under processed/<report_dir>
    try:
        processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
        cfg_dir = getattr(config, 'report_dir', 'reports')
        reports_dir = cfg_dir if os.path.isabs(cfg_dir) else os.path.join(processed_dir, cfg_dir)
        os.makedirs(reports_dir, exist_ok=True)
        dynamics_path = os.path.join(reports_dir, 'training_dynamics.csv')
    except Exception:
        processed_dir = None
        reports_dir = None
        dynamics_path = None
    for epoch in range(config.max_epochs):
        logger.info(f'-------------> Epoch {epoch}')

        # Unfreeze log_sigma after warmup
        if freeze_epochs > 0 and epoch == freeze_epochs:
            for p in [getattr(model, 'log_sigma_structure', None), getattr(model, 'log_sigma_latency', None), getattr(model, 'log_sigma_host', None)]:
                if p is not None:
                    p.requires_grad = True

        # Train 1 epoch
        logger.info(f'Training...')

        train_total, train_struct, train_lat = train_epoch(config, train_loader, model, optimizer)
        # Set output
        logger.info(f'Train Epoch: {epoch}  Loss: {train_total} Structure Loss: {train_struct} Latency Loss: {train_lat}')

        # Append training dynamics (log sigmas and loss) for visualization
        try:
            if dynamics_path:
                s_struc = float(getattr(model, 'log_sigma_structure', 0.0))
                try:
                    s_struc = float(getattr(model, 'log_sigma_structure').item())
                except Exception:
                    pass
                s_lat = float(getattr(model, 'log_sigma_latency', 0.0))
                try:
                    s_lat = float(getattr(model, 'log_sigma_latency').item())
                except Exception:
                    pass
                s_host = float(getattr(model, 'log_sigma_host', 0.0) or 0.0)
                try:
                    if getattr(model, 'log_sigma_host', None) is not None:
                        s_host = float(getattr(model, 'log_sigma_host').item())
                except Exception:
                    pass

                write_header = not os.path.isfile(dynamics_path)
                with open(dynamics_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(['epoch', 'log_sigma_struct', 'log_sigma_lat', 'log_sigma_host', 'loss_total', 'loss_struct', 'loss_lat'])
                    writer.writerow([int(epoch), s_struc, s_lat, s_host, float(train_total), float(train_struct), float(train_lat)])
        except Exception as e:
            logger.debug(f"Skip writing training dynamics CSV: {e}")

        # DUBUG:取消Val，缩短时间
        # # Val 1 epoch
        # val_total, val_struct, val_lat = val_epoch(config, val_loader, model)
        # # Set output
        # logger.info(f'Valid Epoch: {epoch}  Loss: {val_total} Structure Loss: {val_struct} Latency Loss: {val_lat}')

        # If validation loss is smaller than min_val_loss, save the model
        
        # DEBUG:取消Val，缩短时间
        # if val_total < min_val_loss:
        #     min_val_loss = val_total
        if train_total < min_val_loss:
            min_val_loss = train_total

            if test_loader is not None:
                logger.info("Valid loss is smaller. Start evaluation...")
                evaluate(config, test_loader, model, epoch=epoch)

            # Save model to configured path
            try:
                rel_model_path = getattr(config, 'model_path', 'model.pth')
            except Exception:
                rel_model_path = 'model.pth'

            # If path is relative, interpret it under dataset_root_dir/dataset
            if os.path.isabs(rel_model_path):
                model_path = rel_model_path
            else:
                base_dir = os.path.join(config.dataset_root_dir, config.dataset)
                model_path = os.path.join(base_dir, rel_model_path)

            try:
                model_dir = os.path.dirname(model_path)
                if model_dir:
                    os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                logger.info(f"Valid loss is smaller. Model saved to {model_path}.")
            except Exception as e:
                logger.warning(f"Failed to save model to {model_path}: {e}")
        
    # Final evaluation
    if test_loader is not None:
        evaluate(config, test_loader, model, epoch=None)
        
    logger.info("Training finished.")
