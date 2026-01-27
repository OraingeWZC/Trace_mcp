from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from loguru import logger
import yaml
import os


def load_service_id_to_name(yaml_path: str) -> Dict[int, str]:
    """
    从 YAML 文件加载 service_name -> id 映射，并反转为 id -> name。
    假设 YAML 格式为:
        service-a.prod: 0
        service-b.staging: 1
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Service mapping file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        name_to_id = yaml.safe_load(f)
    
    if not isinstance(name_to_id, dict):
        raise ValueError(f"YAML file {yaml_path} does not contain a dictionary.")
    
    # 反转为 id -> name
    id_to_name = {}
    for name, sid in name_to_id.items():
        try:
            id_to_name[int(sid)] = str(name)
        except (ValueError, TypeError):
            logger.warning(f"Skipping invalid entry in YAML: {name}: {sid}")
    
    logger.info(f"Loaded {len(id_to_name)} service_id -> service_name mappings from {yaml_path}")
    return id_to_name


def evaluate_with_root_cause(
    all_trace_info: List[Dict[str, Any]],
    true_root_causes: Dict[str, Any],
    topk: int = 3,
    service_id_to_name: Dict[int, str] = None  # id -> name
) -> Tuple[List[Dict[str, Any]], float, float, float]:
    """
    对每个异常trace，选出分数最高的topk个节点的service_id作为候选根因，
    并与真实根因对比，计算topk准确率。
    
    要求：
      - true_root_causes 中的值是 service_id (int)
      - service_id_to_name 将 service_id 映射为完整服务名字符串（如 "payment-service.prod"）
    """

    def get_service_prefix(service_name: str) -> str:
        """提取服务名的前缀（点号前的部分），若无点号则返回原名"""
        if isinstance(service_name, str) and '.' in service_name:
            return service_name.split('.')[0]
        return service_name

    results = []
    correct_top1 = 0
    correct_top3 = 0
    correct_topk = 0
    total = 0
    anomalous_count = 0
    gt_not_in_graph_count = 0  # 统计真实根因不在图中的数量

    # 构建安全的映射函数
    if service_id_to_name is None:
        logger.warning("No service_id_to_name mapping provided! Using str(id) as fallback.")
        id_to_name_func = lambda x: str(x)
    else:
        id_to_name_func = lambda x: service_id_to_name.get(int(x), f"unknown-{x}")

    for idx, trace in enumerate(all_trace_info):
        trace_id = trace['trace_id']
        is_anomalous = trace['is_anomalous']
        graph = trace['graph']
        node_scores = trace['node_scores'].detach().cpu().numpy()
        service_ids = graph.ndata['service_id'].cpu().numpy()

        if is_anomalous:
            anomalous_count += 1

        if is_anomalous and len(node_scores) > 0:
            node_scores_tensor = torch.from_numpy(node_scores)
            softmax_scores = torch.softmax(node_scores_tensor, dim=0).numpy()

            unique_services = np.unique(service_ids)
            service_scores = {}
            for service in unique_services:
                service_node_indices = np.where(service_ids == service)[0]
                service_score = np.sum(softmax_scores[service_node_indices])
                service_scores[service] = service_score

            sorted_services = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)
            topk_services = [service for service, _ in sorted_services[:topk]]
            topk_scores = [score for _, score in sorted_services[:topk]]
            topk_idx = np.argsort(node_scores)[-topk:][::-1]

            # 保存所有服务的分数用于调试
            all_sorted_services = sorted_services
        else:
            topk_idx = np.argsort(node_scores)[-topk:][::-1]
            topk_services = [service_ids[i] for i in topk_idx]
            topk_scores = [node_scores[i] for i in topk_idx]
            all_sorted_services = None  # 非异常trace不需要

        gt = true_root_causes.get(trace_id, None)
        is_correct_top1 = False
        is_correct_top3 = False
        is_correct_topk = False

        if is_anomalous and gt is None and anomalous_count <= 3:
            logger.warning(f"异常trace未找到真实根因 - trace_id: {trace_id}")

        if is_anomalous and gt is not None:
            total += 1

            # === 关键：将 service_id 转为 name 再取 prefix ===
            try:
                gt_name = id_to_name_func(gt)
                gt_prefix = get_service_prefix(gt_name)
            except Exception as e:
                logger.error(f"Failed to process ground truth {gt}: {e}")
                gt_prefix = str(gt)

            pred_names = [id_to_name_func(s) for s in topk_services]
            pred_prefixes = [get_service_prefix(name) for name in pred_names]

            # 检查真实根因是否在图中
            gt_in_graph = any(sid == gt for sid, _ in all_sorted_services) if all_sorted_services else False
            if not gt_in_graph:
                gt_not_in_graph_count += 1

            if total <= 3:  # 增加到前3个样本
                logger.debug(f"\n{'='*50}")
                logger.debug(f"异常样本 #{total} - 根因分析")
                logger.debug(f"GT: id={gt} → name='{gt_name}' → prefix='{gt_prefix}'")
                logger.debug(f"GT在图中: {'是' if gt_in_graph else '否'}")
                logger.debug(f"Top-{min(3, len(pred_names))} 预测: {pred_names[:3]}")

                # 查找真实根因在所有服务中的排名和分数
                if all_sorted_services is not None:
                    # 先检查真实根因service_id是否在图中
                    if not gt_in_graph:
                        logger.warning(f"⚠ 真实根因 service_id={gt} ('{gt_name}') 不在trace图中！将使用前缀匹配。")

                    # 找到真实根因的分数和排名（按前缀匹配）
                    gt_rank = None
                    gt_score = None
                    matched_service_id = None
                    for rank_idx, (service_id, score) in enumerate(all_sorted_services, 1):
                        service_name = id_to_name_func(service_id)
                        service_prefix = get_service_prefix(service_name)
                        if service_prefix == gt_prefix:
                            gt_rank = rank_idx
                            gt_score = score
                            matched_service_id = service_id
                            break

                    if gt_rank is not None:
                        logger.debug(f"真实根因 '{gt_prefix}' (id={gt}) 排名: #{gt_rank}/{len(all_sorted_services)}, 分数: {gt_score:.4f}")
                        if matched_service_id != gt:
                            logger.debug(f"  注意：匹配的是前缀相同的服务 id={matched_service_id} ('{id_to_name_func(matched_service_id)}')")
                    else:
                        logger.warning(f"真实根因前缀 '{gt_prefix}' 在所有服务中都未找到！")

                    # 显示所有服务的分数（前10个）
                    logger.debug(f"所有服务分数排名（前10）:")
                    for rank_idx, (service_id, score) in enumerate(all_sorted_services[:10], 1):
                        service_name = id_to_name_func(service_id)
                        is_gt = "★" if service_id == gt else " "
                        logger.debug(f"  {is_gt} #{rank_idx}: {service_name} (id={service_id}) → {score:.4f}")

            # Top-1
            if len(pred_prefixes) > 0 and pred_prefixes[0] == gt_prefix:
                correct_top1 += 1
                is_correct_top1 = True

            # Top-3
            if gt_prefix in pred_prefixes[:3]:
                correct_top3 += 1
                is_correct_top3 = True

            # Top-k
            if gt_prefix in pred_prefixes:
                correct_topk += 1
                is_correct_topk = True

        results.append({
            'trace_id': trace_id,
            'is_anomalous': is_anomalous,
            'top_candidates': [(int(service_ids[i]), float(node_scores[i])) for i in topk_idx],
            'groundtruth': gt,
            'is_correct_top1': is_correct_top1,
            'is_correct_top3': is_correct_top3,
            'is_correct_topk': is_correct_topk
        })

    acc_top1 = correct_top1 / total if total > 0 else 0.0
    acc_top3 = correct_top3 / total if total > 0 else 0.0
    acc_topk = correct_topk / total if total > 0 else 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"根因定位统计汇总:")
    logger.info(f"  总异常trace数: {anomalous_count}")
    logger.info(f"  有真实根因标注: {total}")
    logger.info(f"  真实根因不在图中: {gt_not_in_graph_count} ({gt_not_in_graph_count/total*100:.1f}%)" if total > 0 else "  真实根因不在图中: 0")
    logger.info(f"  正确匹配数: Top1={correct_top1}, Top3={correct_top3}, Top{topk}={correct_topk}")
    logger.info(f"  准确率: Top1={acc_top1:.4f}, Top3={acc_top3:.4f}, Top{topk}={acc_topk:.4f}")
    logger.info(f"{'='*60}")

    if gt_not_in_graph_count > 0:
        logger.warning(f"\n⚠ 发现 {gt_not_in_graph_count}/{total} 个样本的真实根因不在trace图中！")
        logger.warning(f"这可能是因为:")
        logger.warning(f"  1. root_cause 使用的是故障注入时的service名，但trace中使用的是运行时的service名")
        logger.warning(f"  2. service_id映射不一致（如 'shipping.memory' vs 'shipping'）")
        logger.warning(f"  3. 异常导致相关服务的span没有被采集")
        logger.warning(f"建议: 检查数据集的root_cause标注和service_id映射是否一致\n")

    return results, acc_top1, acc_top3, acc_topk