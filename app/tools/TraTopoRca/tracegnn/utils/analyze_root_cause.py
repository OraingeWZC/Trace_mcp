from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from loguru import logger

def evaluate_with_root_cause(
    all_trace_info: List[Dict[str, Any]],
    true_root_causes: Dict[str, Any],
    topk: int = 5,
    svc_pool_k: int = 3,
    svc_len_p: float = 0.25,
    svc_tau: float = 0.8,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    对每个异常trace，选出分数最高的topk个节点的service_id作为候选根因，
    并与真实根因对比，计算topk准确率。
    """
    results = []
    correct_top1 = 0
    correct_topk = 0
    total = 0

    for trace in all_trace_info:
        trace_id = trace['trace_id']
        is_anomalous = trace['is_anomalous']
        graph = trace['graph']
        node_scores = trace['node_scores'].detach().cpu().numpy()
        # 获取每个节点的service_id
        service_ids = graph.ndata['service_id'].cpu().numpy()

        # 两段式服务聚合：先服务内 Top-K 池化 + 节点数校正，再跨服务温度 Softmax
        if is_anomalous and len(node_scores) > 0:
            unique_services = np.unique(service_ids)
            # 可调参数：服务内池化K、节点数校正指数p、温度tau（默认值见函数参数）
            p = float(svc_len_p)
            tau = float(svc_tau)

            # 1) 服务内：sum_topK / n_s^p
            per_service_scores: Dict[int, float] = {}
            for s in unique_services:
                idx = np.where(service_ids == s)[0]
                n_s = int(len(idx))
                if n_s == 0:
                    per_service_scores[int(s)] = 0.0
                    continue
                k = min(svc_pool_k, n_s)
                vals = node_scores[idx]
                # 选服务内 Top-K 节点分数求和
                if k > 0:
                    # 更稳健的 Top-K：使用分区而非全排序
                    part = np.partition(vals, -k)[-k:]
                    sum_topk = float(np.sum(part))
                else:
                    sum_topk = 0.0
                per_service_scores[int(s)] = sum_topk / (max(1.0, float(n_s)) ** p)

            # 2) 跨服务：Softmax(score_s / tau)
            if per_service_scores:
                services = list(per_service_scores.keys())
                score_vec = torch.tensor([per_service_scores[s] for s in services], dtype=torch.float32)
                probs = torch.softmax(score_vec / tau, dim=0).detach().cpu().numpy()
                service_rank = sorted(zip(services, probs), key=lambda x: x[1], reverse=True)
                topk_services = [int(s) for s, _ in service_rank[:topk]]
                topk_scores = [float(pv) for _, pv in service_rank[:topk]]
            else:
                topk_services, topk_scores = [], []

            # 兼容显示：节点级 Top-K（不参与服务排序）
            topk_idx = np.argsort(node_scores)[-topk:][::-1]
        else:
            # 退化：按节点排序近似服务排序
            topk_idx = np.argsort(node_scores)[-topk:][::-1]
            topk_services = [int(service_ids[i]) for i in topk_idx]
            topk_scores = [float(node_scores[i]) for i in topk_idx]

        # logger.debug(f"Trace {trace_id}: top{topk} node idx={topk_idx}, service_ids={topk_services}, scores={topk_scores}")

        # 真实根因
        gt = true_root_causes.get(trace_id, None)
        is_correct_top1 = False
        is_correct_topk = False
        if is_anomalous and gt is not None:
            total += 1
            if topk_services[0] == gt:
                correct_top1 += 1
                is_correct_top1 = True
            if gt in topk_services:
                correct_topk += 1
                is_correct_topk = True

        results.append({
            'trace_id': trace_id,
            'is_anomalous': is_anomalous,
            'top_candidates': [(int(service_ids[i]), float(node_scores[i])) for i in topk_idx],
            'groundtruth': gt,
            'is_correct_top1': is_correct_top1,
            'is_correct_topk': is_correct_topk
        })

        # logger.debug(f"Trace {trace_id}: groundtruth={gt}, top1={topk_services[0] if len(topk_services)>0 else None}, "
        #              f"topk={topk_services}, is_correct_top1={is_correct_top1}, is_correct_topk={is_correct_topk}")

    acc_top1 = correct_top1 / total if total > 0 else 0.0
    acc_topk = correct_topk / total if total > 0 else 0.0
    # logger.info(f"Top1根因定位准确率: {acc_top1:.4f}，Top{topk}根因定位准确率: {acc_topk:.4f}，样本数: {total}")
    return results, acc_top1, acc_topk


def evaluate_mixed_root_cause_joint(
    all_trace_info: List[Dict[str, Any]],
    true_root_causes: Dict[str, Any],
    topk: int = 5,
    svc_pool_k: int = 3,
    svc_len_p: float = 0.25,
    host_nll_key: str = 'host_nll',
    lambda_host: float = 0.35,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], float, float, float, float, float, float]:
    """
    混合（服务+主机）联合评分评估：
    - 服务侧：对节点分数进行服务内Top-K池化+节点数惩罚，得到 per-service 分数并鲁棒标准化（median/MAD）。
    - 主机侧：从 trace_info 中读取 host_nll（每host的NLL），鲁棒标准化。
    - 统一融合：混合榜单中，服务项赋权 (1-lambda_host)*svc_z，主机项赋权 lambda_host*host_z。
    - 不依赖拓扑/URS/SInfra，仅基于当前trace的信息，适合“无拓扑，训练期”场景。
    返回：results 列表与 Top1/TopK 的总体、服务子集、主机子集准确率。
    """
    import numpy as _np

    def _robust_norm(d: Dict[int, float]) -> Dict[int, float]:
        if not d:
            return {}
        arr = _np.array(list(d.values()), dtype=_np.float64)
        med = float(_np.median(arr))
        mad = float(_np.median(_np.abs(arr - med)))
        denom = mad if mad > 1e-6 else float(_np.std(arr))
        if denom <= 1e-6:
            denom = 1.0
        return {int(k): (float(v) - med) / denom for k, v in d.items()}

    results: List[Dict[str, Any]] = []
    total = top1 = topk_correct = 0
    svc_total = svc_top1 = svc_topk = 0
    host_total = host_top1 = host_topk = 0

    for trace in all_trace_info:
        if not trace.get('is_anomalous', False):
            continue
        trace_id = trace['trace_id']
        g = trace['graph']
        node_scores = trace['node_scores'].detach().cpu().numpy()
        service_ids = g.ndata['service_id'].detach().cpu().numpy()
        host_ids = g.ndata['host_id'].detach().cpu().numpy()

        # 真实根因
        gt = true_root_causes.get(trace_id, None)
        try:
            gt_int = int(gt) if gt is not None else 0
        except Exception:
            gt_int = 0
        if gt_int <= 0:
            continue

        # 是否为主机类故障（若无法解析类别名，回退到“gt是否在host集合内”）
        try:
            is_host_fault = gt_int in set(map(int, host_ids.tolist()))
        except Exception:
            is_host_fault = False

        # 1) 服务侧聚合：sum_topK / n_s^p
        unique_services = _np.unique(service_ids)
        p = float(svc_len_p)
        per_service_scores: Dict[int, float] = {}
        for s in unique_services:
            idx = _np.where(service_ids == s)[0]
            n_s = int(len(idx))
            if n_s == 0:
                per_service_scores[int(s)] = 0.0
                continue
            k = min(svc_pool_k, n_s)
            if k > 0:
                vals = node_scores[idx]
                part = _np.partition(vals, -k)[-k:]
                sum_topk = float(_np.sum(part))
            else:
                sum_topk = 0.0
            per_service_scores[int(s)] = sum_topk / (max(1.0, float(n_s)) ** p)
        svc_n = _robust_norm(per_service_scores)

        # 2) 主机侧：HostNLL（若无，则主机候选为空）
        host_nll = trace.get(host_nll_key, None)
        host_n = _robust_norm(host_nll) if isinstance(host_nll, dict) and host_nll else {}

        # 3) 统一混合榜：同一尺度上融合
        w = float(lambda_host)
        mixed: List[Tuple[str, int, float]] = []
        for s, v in svc_n.items():
            mixed.append(('svc', int(s), (1.0 - w) * float(v)))
        for h, v in host_n.items():
            mixed.append(('host', int(h), w * float(v)))

        mixed.sort(key=lambda x: x[2], reverse=True)
        top = mixed[:topk]
        kinds = [k for (k, i, s) in top]
        ids = [i for (k, i, s) in top]

        rec = {
            'trace_id': trace_id,
            'groundtruth': gt,
            'top_mixed': top,
            'is_correct_top1': False,
            'is_correct_topk': False,
            'is_host_fault': is_host_fault,
        }

        total += 1
        if is_host_fault:
            host_total += 1
            if kinds and kinds[0] == 'host' and int(ids[0]) == gt_int:
                top1 += 1; host_top1 += 1; rec['is_correct_top1'] = True
            ok = any(k == 'host' and int(i) == gt_int for (k, i, _) in top)
            if ok:
                topk_correct += 1; host_topk += 1; rec['is_correct_topk'] = True
        else:
            svc_total += 1
            if kinds and kinds[0] == 'svc' and int(ids[0]) == gt_int:
                top1 += 1; svc_top1 += 1; rec['is_correct_top1'] = True
            ok = any(k == 'svc' and int(i) == gt_int for (k, i, _) in top)
            if ok:
                topk_correct += 1; svc_topk += 1; rec['is_correct_topk'] = True

        results.append(rec)

    acc_top1 = (top1 / total) if total > 0 else 0.0
    acc_topk = (topk_correct / total) if total > 0 else 0.0
    acc_svc_top1 = (svc_top1 / svc_total) if svc_total > 0 else 0.0
    acc_svc_topk = (svc_topk / svc_total) if svc_total > 0 else 0.0
    acc_host_top1 = (host_top1 / host_total) if host_total > 0 else 0.0
    acc_host_topk = (host_topk / host_total) if host_total > 0 else 0.0
    return results, acc_top1, acc_topk, acc_svc_top1, acc_svc_topk, acc_host_top1, acc_host_topk
