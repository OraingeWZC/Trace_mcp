from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np
import torch
from loguru import logger
import os
import csv
from datetime import datetime

from tracegnn.data.trace_graph import TraceGraphIDManager
from .sinfra_core import (
    _minute_key_from_graph,
    _robust_normalize,
    compute_sinfra_per_host,
)
from .host_eval import _host_pool_scores


def _service_pool_scores(node_scores: np.ndarray,
                         service_ids: np.ndarray,
                         topk_pool: int = 2) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = {}
    for i in range(len(node_scores)):
        s = int(service_ids[i])
        if s <= 0:
            continue
        buckets.setdefault(s, []).append(float(node_scores[i]))
    pooled: Dict[int, float] = {}
    for s, scores in buckets.items():
        if not scores:
            continue
        scores.sort(reverse=True)
        pooled[s] = float(sum(scores[:max(1, int(topk_pool))]))
    return pooled


def _apply_volatility_pruning(host_scores: Dict[int, float],
                              trace_graph,
                              threshold: float = 0.01) -> Dict[int, float]:
    """
    Volatility pruning for host candidates: if a host's recent metric window
    shows extremely low variability, down-weight its score as likely background noise.
    Expects optional attribute `trace_graph.host_seq` attached by dataset loader
    (a mapping: host_id -> Tensor[W, D]).
    """
    try:
        host_seq_map = getattr(trace_graph, 'host_seq', {})
    except Exception:
        host_seq_map = {}
    if not host_seq_map:
        return host_scores
    pruned = dict(host_scores)
    for hid, sc in list(pruned.items()):
        try:
            seq = host_seq_map.get(int(hid))
            if seq is None:
                continue
            arr = seq.detach().cpu().numpy() if hasattr(seq, 'detach') else seq
            if arr is None:
                continue
            std_vals = np.std(arr, axis=0)
            max_vol = float(np.max(std_vals)) if std_vals.size > 0 else 0.0
            if max_vol < float(threshold):
                pruned[hid] = float(sc) * 0.3
        except Exception:
            # be robust to malformed host_seq entries
            pass
    return pruned


def _apply_hetero_propagation(svc_scores: Dict[int, float],
                              host_scores: Dict[int, float],
                              service_ids: np.ndarray,
                              host_ids: np.ndarray,
                              beta: float = 0.3) -> Dict[int, float]:
    """
    Heterogeneous propagation from host to service: for each (service, host)
    co-located pair in the trace, add beta * host_score to the service.
    """
    try:
        s_np = service_ids.detach().cpu().numpy()
        h_np = host_ids.detach().cpu().numpy()
    except Exception:
        return svc_scores
    svc_to_host: Dict[int, int] = {}
    try:
        for i in range(len(s_np)):
            s = int(s_np[i]); h = int(h_np[i])
            if s > 0 and h > 0:
                svc_to_host.setdefault(s, h)
    except Exception:
        pass
    if not svc_to_host:
        return svc_scores
    out = dict(svc_scores)
    for s, v in list(out.items()):
        hid = svc_to_host.get(int(s))
        if hid is not None and int(hid) in host_scores:
            h_sc = float(host_scores[int(hid)])
            if h_sc > 0.0 and beta and beta > 0:
                out[int(s)] = float(v) + float(beta) * h_sc
    return out


# ============================
# URS 路径（基础 + SInfra + Jaccard）
# ============================

def evaluate_mixed_root_cause_urs(
    all_trace_info: List[Dict[str, Any]],
    true_root_causes: Dict[str, Any],
    host_adj: Dict[int, Set[int]],
    topk: int = 5,
    svc_pool_k: int = 2,
    host_pool_k: int = 3,
    alpha: float = 0.7,
    steps: int = 1,
    fault_category_names: Optional[Dict[int, str]] = None,
    lambda_host: float = 1.0,
    id_manager: Optional[TraceGraphIDManager] = None,
    infra_index: Optional[Dict[str, Dict[str, Any]]] = None,
    W: int = 3,
    urs_alpha: float = 0.6,
    debug: bool = False,
    ms_W_list: Optional[List[int]] = None,
    lse_tau: float = 2.0,
    sinfra_w: Optional[Dict[str, float]] = None,
    peer_mode: str = 'trace',
    host_nll_key: Optional[str] = None,
    # Cross-modal truncation and topology boost
    mixed_pool_svc_k: int = 3,
    mixed_pool_host_k: int = 3,
    topo_boost_beta: float = 0.2,
    # Optional debug export location
    conflict_csv_path: Optional[str] = None,
    epoch: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float, float, float, float, float, float]:
    results: List[Dict[str, Any]] = []
    total = top1 = topk_correct = 0
    svc_total = svc_top1 = svc_topk = 0
    host_total = host_top1 = host_topk = 0

    host_name: Dict[int, str] = {}
    if id_manager is not None:
        try:
            host_name = {i: id_manager.host_id.rev(i) for i in range(len(id_manager.host_id))}
        except Exception:
            host_name = {}

    # 1) 统计全局的 service-host 共现对，计算 Jaccard 先验
    pair_counts: Dict[Tuple[int, int], Dict[str, int]] = {}
    Nfail = sum(1 for t in all_trace_info if t.get('is_anomalous', False))

    def pairs_in_graph(g) -> Set[Tuple[int, int]]:
        sids = g.ndata['service_id'].detach().cpu().numpy()
        hids = g.ndata['host_id'].detach().cpu().numpy()
        ps: Set[Tuple[int, int]] = set()
        for i in range(len(sids)):
            h = int(hids[i]); s = int(sids[i])
            if h > 0 and s > 0:
                ps.add((s, h))
        return ps

    for trace in all_trace_info:
        g = trace['graph']
        P = pairs_in_graph(g)
        if not P:
            continue
        if trace.get('is_anomalous', False):
            for p in P:
                pc = pair_counts.setdefault(p, {'ef': 0, 'ep': 0, 'nf': 0}); pc['ef'] += 1
        else:
            for p in P:
                pc = pair_counts.setdefault(p, {'ef': 0, 'ep': 0, 'nf': 0}); pc['ep'] += 1
    for p, pc in pair_counts.items():
        pc['nf'] = max(Nfail - pc['ef'], 0)

    jaccard = {p: (pc['ef'] / max(pc['ef'] + pc['ep'] + pc['nf'], 1e-6)) for p, pc in pair_counts.items()}
    nonzero_j = sum(1 for _, v in jaccard.items() if v > 0.0)
    pair_nonzero_ratio = (nonzero_j / max(len(jaccard), 1)) if jaccard else 0.0

    # 2) 针对每条异常 trace 进行 URS 混合排序
    for trace in all_trace_info:
        if not trace.get('is_anomalous', False):
            continue
        trace_id = trace['trace_id']
        gt = true_root_causes.get(trace_id, None)
        try:
            gt_int = int(gt) if gt is not None else 0
        except Exception:
            gt_int = 0
        if gt_int <= 0:
            continue

        g = trace['graph']
        service_ids = g.ndata['service_id']
        host_ids = g.ndata['host_id']
        fc_id = trace.get('fault_category', None)

        # 识别是否主机类故障
        is_host_fault = False
        if fc_id is not None and fault_category_names:
            name = fault_category_names.get(int(fc_id), '').lower()
            is_host_fault = name.startswith('node') or ('host' in name)
        else:
            try:
                is_host_fault = gt_int in set(map(int, host_ids.detach().cpu().numpy().tolist()))
            except Exception:
                is_host_fault = False

        # Host SInfra + HostNLL 融合
        t0_min_ms = _minute_key_from_graph(g)
        uniq_hosts = sorted({int(h) for h in host_ids.detach().cpu().numpy() if int(h) > 0})
        sinfra: Dict[int, float] = {}
        if infra_index is not None and t0_min_ms is not None and id_manager is not None:
            peer_hosts_names: List[str] = []
            try:
                peer_hosts_names = [host_name.get(h, None) for h in uniq_hosts if host_name.get(h, None)]
            except Exception:
                peer_hosts_names = []
            for hid in uniq_hosts:
                hnm = host_name.get(hid, None)
                if not hnm:
                    continue
                v = compute_sinfra_per_host(
                    hnm, infra_index, t0_min_ms, W=W,
                    ms_W_list=ms_W_list, lse_tau=lse_tau, sinfra_w=sinfra_w,
                    peer_mode=peer_mode, peer_hosts=peer_hosts_names,
                )
                if v is not None:
                    sinfra[hid] = float(v)
        sinfra_n = _robust_normalize(sinfra) if sinfra else {}

        host_nll = trace.get(host_nll_key, None) if host_nll_key else None
        if isinstance(host_nll, dict) and host_nll:
            arr = np.array(list(host_nll.values()), dtype=np.float64)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            denom = mad if mad > 1e-6 else float(np.std(arr))
            denom = denom if denom > 1e-6 else 1.0
            host_nll_norm = {int(k): (float(v) - med) / denom for k, v in host_nll.items()}
            w = float(lambda_host)
            for hid in uniq_hosts:
                s_sinfra = float(sinfra_n.get(hid, 0.0))
                s_nll = float(host_nll_norm.get(hid, 0.0))
                sinfra_n[hid] = w * s_nll + (1.0 - w) * s_sinfra

        # 计算每对 (s,h) 的 URS 分数
        P = set()
        sids = service_ids.detach().cpu().numpy(); hids = host_ids.detach().cpu().numpy()
        for i in range(len(sids)):
            h = int(hids[i]); s = int(sids[i])
            if h > 0 and s > 0:
                P.add((s, h))

        urs_pairs: List[Tuple[Tuple[int, int], float]] = []
        for (s, h) in P:
            ji = float(jaccard.get((s, h), 0.0))
            pri = float(sinfra_n.get(h, 0.0))
            urs = urs_alpha * ji + (1.0 - urs_alpha) * pri
            urs_pairs.append(((s, h), urs))
        urs_pairs.sort(key=lambda x: x[1], reverse=True)

        svc_best: Dict[int, float] = {}
        host_best: Dict[int, float] = {}
        for (s, h), sc in urs_pairs:
            svc_best[s] = max(svc_best.get(s, float('-inf')), float(sc))
            host_best[h] = max(host_best.get(h, float('-inf')), float(sc))

        # Topology boost: service receives bonus from its related high-score hosts
        if topo_boost_beta and topo_boost_beta > 0:
            try:
                sids_np = service_ids.detach().cpu().numpy(); hids_np = host_ids.detach().cpu().numpy()
                svc_to_hosts: Dict[int, Set[int]] = {}
                for i in range(len(sids_np)):
                    s = int(sids_np[i]); h = int(hids_np[i])
                    if s > 0 and h > 0:
                        svc_to_hosts.setdefault(s, set()).add(h)
                for s_id in list(svc_best.keys()):
                    rel_hosts = svc_to_hosts.get(int(s_id), set())
                    max_h = 0.0
                    for h_id in rel_hosts:
                        if h_id in host_best:
                            max_h = max(max_h, float(host_best[h_id]))
                    if max_h > 0.0:
                        svc_best[s_id] = float(svc_best[s_id]) + float(topo_boost_beta) * max_h
            except Exception:
                pass

        # Cross-modal truncation
        if isinstance(mixed_pool_svc_k, int) and mixed_pool_svc_k > 0 and len(svc_best) > mixed_pool_svc_k:
            svc_best = dict(sorted(svc_best.items(), key=lambda x: x[1], reverse=True)[:mixed_pool_svc_k])
        if isinstance(mixed_pool_host_k, int) and mixed_pool_host_k > 0 and len(host_best) > mixed_pool_host_k:
            host_best = dict(sorted(host_best.items(), key=lambda x: x[1], reverse=True)[:mixed_pool_host_k])

        svc_rk = sorted(svc_best.items(), key=lambda kv: kv[1], reverse=True)
        host_rk = sorted(host_best.items(), key=lambda kv: kv[1], reverse=True)

        mixed: List[Tuple[str, int, float]] = []
        mixed.extend([('svc', int(s), float(v)) for s, v in svc_rk])
        mixed.extend([('host', int(h), float(v)) for h, v in host_rk])
        mixed.sort(key=lambda x: x[2], reverse=True)

        top = mixed[:topk]
        kinds = [k for (k, i, s) in top]
        ids = [i for (k, i, s) in top]

        rec = {
            'trace_id': trace_id,
            'groundtruth': gt,
            'fault_category': fc_id,
            'top_mixed': top,
            'is_correct_top1': False,
            'is_correct_topk': False,
            'is_host_fault': is_host_fault,
        }

        # Score scale diagnostics CSV (svc/host maxima per-trace)
        # Saved next to conflict CSV if available; otherwise in CWD
        try:
            _svc_max_for_debug = max(svc_best.values()) if svc_best else 0.0
            _host_max_for_debug = max(host_best.values()) if host_best else 0.0
            dbg_dir = os.path.dirname(conflict_csv_path) if conflict_csv_path else None
            dbg_path = os.path.join(dbg_dir, 'score_distribution_debug.csv') if dbg_dir else 'score_distribution_debug.csv'
            write_header = not os.path.isfile(dbg_path)
            with open(dbg_path, 'a', newline='', encoding='utf-8') as dfp:
                w = csv.writer(dfp)
                if write_header:
                    w.writerow(['timestamp','epoch','trace_id','gt_type','svc_max','host_max','lambda_host'])
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                gt_type = 'host' if is_host_fault else 'svc'
                w.writerow([ts, (int(epoch) if epoch is not None else ''), str(trace_id), gt_type, float(_svc_max_for_debug), float(_host_max_for_debug), float(lambda_host)])
        except Exception:
            pass

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

            # Debug: 捕获“服务为真但被主机抢Top-1”的冲突样例（base+URS路径）
            if debug and top:
                k0, i0, s0 = top[0]
                if k0 == 'host':
                    gt_rank = -1
                    gt_score = float('nan')
                    for idx_m, (km, im, sm) in enumerate(mixed):
                        if km == 'svc' and int(im) == int(gt_int):
                            gt_rank = idx_m + 1
                            gt_score = float(sm)
                            break
                    if gt_rank > 0:
                        delta = float(s0) - float(gt_score)
                        logger.info(f"[RCA Conflict] trace={trace_id} gt_svc={gt_int} gt_rank={gt_rank} gt_score={gt_score:.4f} top1_host={i0} top1_score={s0:.4f} delta={delta:.4f}")
                        try:
                            if conflict_csv_path:
                                write_header = not os.path.isfile(conflict_csv_path)
                                with open(conflict_csv_path, 'a', newline='', encoding='utf-8') as cf:
                                    w = csv.writer(cf)
                                    if write_header:
                                        w.writerow(['timestamp','epoch','trace_id','is_host_fault','groundtruth','gt_int','gt_rank','gt_score','top1_kind','top1_id','top1_score','delta'])
                                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    w.writerow([ts, (int(epoch) if epoch is not None else ''), str(trace_id), False, gt, int(gt_int), int(gt_rank), float(gt_score), str(k0), int(i0), float(s0), float(delta)])
                        except Exception:
                            pass

        results.append(rec)

    acc_top1 = (top1 / total) if total > 0 else 0.0
    acc_topk = (topk_correct / total) if total > 0 else 0.0
    acc_svc_top1 = (svc_top1 / svc_total) if svc_total > 0 else 0.0
    acc_svc_topk = (svc_topk / svc_total) if svc_total > 0 else 0.0
    acc_host_top1 = (host_top1 / host_total) if host_total > 0 else 0.0
    acc_host_topk = (host_topk / host_total) if host_total > 0 else 0.0
    return results, acc_top1, acc_topk, acc_svc_top1, acc_svc_topk, acc_host_top1, acc_host_topk


# ============================
# rerank 路径（服务/主机各自池化 + URS boost）
# ============================

def evaluate_mixed_root_cause_rerank(
    all_trace_info: List[Dict[str, Any]],
    true_root_causes: Dict[str, Any],
    id_manager: Optional[TraceGraphIDManager] = None,
    infra_index: Optional[Dict[str, Dict[str, Any]]] = None,
    topk: int = 5,
    svc_pool_k: int = 2,
    svc_len_p: float = 0.25,
    svc_tau: float = 0.8,
    host_W: int = 3,
    ms_W_list: Optional[List[int]] = None,
    lse_tau: float = 2.0,
    sinfra_w: Optional[Dict[str, float]] = None,
    eta: float = 0.3,
    lambda_host: float = 0.5,
    rho_mode: str = 'count',
    urs_alpha: float = 0.6,
    gamma_svc: float = 0.3,
    gamma_host: float = 0.3,
    host_nll_key: Optional[str] = None,
    peer_mode: str = 'trace',
    w_jaccard: float = 0.2,
    debug_conflict: bool = False,
    conflict_csv_path: Optional[str] = None,
    epoch: Optional[int] = None,
    # Cross-modal truncation and topology boost
    mixed_pool_svc_k: int = 3,
    mixed_pool_host_k: int = 3,
    topo_boost_beta: float = 0.2,
    # New: hetero propagation and causal pruning
    enable_hetero_prop: bool = False,
    enable_causal_pruning: bool = False,
    pruning_threshold: float = 0.01,
) -> Tuple[List[Dict[str, Any]], float, float, float, float, float, float]:
    results: List[Dict[str, Any]] = []
    total = top1 = topk_correct = 0
    svc_total = svc_top1 = svc_topk = 0
    host_total = host_top1 = host_topk = 0

    host_name: Dict[int, str] = {}
    if id_manager is not None:
        try:
            host_name = {i: id_manager.host_id.rev(i) for i in range(len(id_manager.host_id))}
        except Exception:
            host_name = {}

    # 预计算 Jaccard 对（同 URS）
    pair_counts: Dict[Tuple[int, int], Dict[str, int]] = {}
    Nfail = sum(1 for t in all_trace_info if t.get('is_anomalous', False))
    def pairs_in_graph(g) -> Set[Tuple[int, int]]:
        sids = g.ndata['service_id'].detach().cpu().numpy()
        hids = g.ndata['host_id'].detach().cpu().numpy()
        ps: Set[Tuple[int, int]] = set()
        for i in range(len(sids)):
            h = int(hids[i]); s = int(sids[i])
            if h > 0 and s > 0:
                ps.add((s, h))
        return ps
    for trace in all_trace_info:
        g = trace['graph']
        P = pairs_in_graph(g)
        if not P:
            continue
        if trace.get('is_anomalous', False):
            for p in P:
                pc = pair_counts.setdefault(p, {'ef': 0, 'ep': 0, 'nf': 0}); pc['ef'] += 1
        else:
            for p in P:
                pc = pair_counts.setdefault(p, {'ef': 0, 'ep': 0, 'nf': 0}); pc['ep'] += 1
    for p, pc in pair_counts.items():
        pc['nf'] = max(Nfail - pc['ef'], 0)
    jaccard = {p: (pc['ef'] / max(pc['ef'] + pc['ep'] + pc['nf'], 1e-6)) for p, pc in pair_counts.items()}

    # 主循环
    for trace in all_trace_info:
        if not trace.get('is_anomalous', False):
            continue
        trace_id = trace['trace_id']
        gt = true_root_causes.get(trace_id, None)
        try:
            gt_int = int(gt) if gt is not None else 0
        except Exception:
            gt_int = 0
        if gt_int <= 0:
            continue

        g = trace['graph']
        service_ids = g.ndata['service_id']
        host_ids = g.ndata['host_id']

        # 1) 服务侧基础分：按 service 池化 top-k，并做长度/softmax 归一
        ns = g.ndata['nll'].detach().cpu().numpy() if 'nll' in g.ndata else np.ones_like(service_ids.detach().cpu().numpy())
        svc_base_raw = _service_pool_scores(ns, service_ids.detach().cpu().numpy(), topk_pool=int(svc_pool_k))
        if svc_base_raw:
            services = list(svc_base_raw.keys())
            score_vec = np.array([svc_base_raw[s] / (max(1.0, float((service_ids == s).sum().item())) ** float(svc_len_p)) for s in services], dtype=np.float64)
            # softmax 温度
            probs = np.exp(score_vec / float(svc_tau))
            probs = probs / max(probs.sum(), 1e-6)
            svc_base = {int(s): float(p) for s, p in zip(services, probs)}
        else:
            svc_base = {}

        # 2) 主机侧基础分：模型池化 + rho 注意力 + SInfra (+ HostNLL)
        ns2 = ns
        hids_np = host_ids.detach().cpu().numpy()
        pooled_model = _host_pool_scores(ns2, hids_np, topk_pool=3)
        uniq_hosts = sorted({int(h) for h in hids_np if int(h) > 0})
        # rho 注意力
        rho: Dict[int, float] = {}
        if uniq_hosts:
            if rho_mode == 'duration' and 'latency' in g.ndata and 'span_count' in g.ndata:
                lat = g.ndata['latency'].detach().cpu().numpy()
                spc = g.ndata['span_count'].detach().cpu().numpy()
                val: Dict[int, float] = {}
                for i in range(len(hids_np)):
                    hid = int(hids_np[i]);
                    if hid <= 0:
                        continue
                    val[hid] = val.get(hid, 0.0) + float(lat[i]) * float(spc[i])
            else:
                spc = g.ndata['span_count'].detach().cpu().numpy() if 'span_count' in g.ndata else np.ones_like(hids_np)
                val = {}
                for i in range(len(hids_np)):
                    hid = int(hids_np[i]);
                    if hid <= 0:
                        continue
                    val[hid] = val.get(hid, 0.0) + float(spc[i])
            v = np.array([val.get(h, 0.0) for h in uniq_hosts], dtype=np.float64)
            if np.all(v == 0):
                v = np.ones_like(v)
            v = np.exp(v - v.max()); v = v / max(v.sum(), 1e-6)
            rho = {h: float(v[i]) for i, h in enumerate(uniq_hosts)}

        # SInfra
        sinfra: Dict[int, float] = {}
        if infra_index is not None and id_manager is not None:
            t0_min_ms = _minute_key_from_graph(g)
            if t0_min_ms is not None:
                try:
                    peer_hosts_names = [host_name.get(h, None) for h in uniq_hosts if host_name.get(h, None)]
                except Exception:
                    peer_hosts_names = []
                for hid in uniq_hosts:
                    hnm = host_name.get(hid, None)
                    if not hnm:
                        continue
                    v = compute_sinfra_per_host(
                        hnm, infra_index, t0_min_ms, W=host_W,
                        ms_W_list=ms_W_list, lse_tau=lse_tau, sinfra_w=sinfra_w,
                        peer_mode=peer_mode, peer_hosts=peer_hosts_names,
                    )
                    if v is not None:
                        sinfra[hid] = float(v)
        sinfra_attn = {h: float(v) * (1.0 + float(eta) * float(rho.get(h, 0.0))) for h, v in sinfra.items()} if sinfra else {}
        sinfra_n = _robust_normalize(sinfra_attn) if sinfra_attn else {}

        # 可选：融合 HostNLL（中位数/MAD 标准化）
        host_nll = trace.get(host_nll_key, None) if host_nll_key else None
        if isinstance(host_nll, dict) and host_nll:
            arr = np.array(list(host_nll.values()), dtype=np.float64)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            denom = mad if mad > 1e-6 else float(np.std(arr))
            denom = denom if denom > 1e-6 else 1.0
            host_nll_norm = {int(k): (float(v) - med) / denom for k, v in host_nll.items()}
            w = float(lambda_host)
            for hid in uniq_hosts:
                s_sinfra = float(sinfra_n.get(hid, 0.0))
                s_nll = float(host_nll_norm.get(hid, 0.0))
                sinfra_n[hid] = w * s_nll + (1.0 - w) * s_sinfra

        # 3) 计算 URS boost（用于服务/主机最终分数叠加）
        P = pairs_in_graph(g)
        urs_pairs: List[Tuple[Tuple[int, int], float]] = []
        for (s, h) in P:
            ji = float(jaccard.get((s, h), 0.0))
            pri = float(sinfra_n.get(h, 0.0))
            urs = urs_alpha * ji + (1.0 - urs_alpha) * pri
            urs_pairs.append(((s, h), urs))
        urs_pairs.sort(key=lambda x: x[1], reverse=True)

        # 每个 service/host 的 boost 取与之相关的最大 URS
        svc_boost: Dict[int, float] = {}
        host_boost: Dict[int, float] = {}
        for (s, h), sc in urs_pairs:
            svc_boost[s] = max(svc_boost.get(s, float('-inf')), float(sc))
            host_boost[h] = max(host_boost.get(h, float('-inf')), float(sc))

        # 最终分数：base + gamma * boost
        svc_final = {int(s): float(svc_base.get(s, 0.0)) + float(gamma_svc) * float(svc_boost.get(s, 0.0)) for s in svc_base.keys()}
        host_final = {int(h): float(pooled_model.get(h, 0.0)) + float(gamma_host) * float(host_boost.get(h, 0.0)) for h in pooled_model.keys()}

        # 可选：因果/波动性剪枝（针对 host）
        if enable_causal_pruning:
            try:
                host_final = _apply_volatility_pruning(host_final, g, threshold=float(pruning_threshold))
            except Exception:
                pass

        # 可选：异构传播（host -> service）
        if enable_hetero_prop and (topo_boost_beta and topo_boost_beta > 0):
            try:
                svc_final = _apply_hetero_propagation(svc_final, host_final, service_ids, host_ids, beta=float(topo_boost_beta))
            except Exception:
                pass

        # Cross-modal truncation
        if isinstance(mixed_pool_svc_k, int) and mixed_pool_svc_k > 0 and len(svc_final) > mixed_pool_svc_k:
            svc_final = dict(sorted(svc_final.items(), key=lambda x: x[1], reverse=True)[:mixed_pool_svc_k])
        if isinstance(mixed_pool_host_k, int) and mixed_pool_host_k > 0 and len(host_final) > mixed_pool_host_k:
            host_final = dict(sorted(host_final.items(), key=lambda x: x[1], reverse=True)[:mixed_pool_host_k])

        mixed: List[Tuple[str, int, float]] = []
        mixed.extend([('svc', int(s), float(v)) for s, v in svc_final.items()])
        mixed.extend([('host', int(h), float(v)) for h, v in host_final.items()])
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
        }

        # 统计指标
        total += 1
        # 判断是否主机类 GT（仅用于分桶统计）
        try:
            is_host_fault_k = gt_int in set(map(int, host_ids.detach().cpu().numpy().tolist()))
        except Exception:
            is_host_fault_k = False

        if is_host_fault_k:
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

            # Debug 冲突导出
            if debug_conflict and top:
                k0, i0, s0 = top[0]
                if k0 == 'host':
                    gt_rank = -1
                    gt_score = float('nan')
                    for idx_m, (km, im, sm) in enumerate(mixed):
                        if km == 'svc' and int(im) == int(gt_int):
                            gt_rank = idx_m + 1
                            gt_score = float(sm)
                            break
                    if gt_rank > 0:
                        delta = float(s0) - float(gt_score)
                        logger.info(f"[RCA Conflict] trace={trace_id} gt_svc={gt_int} gt_rank={gt_rank} gt_score={gt_score:.4f} top1_host={i0} top1_score={s0:.4f} delta={delta:.4f}")
                        try:
                            if conflict_csv_path:
                                write_header = not os.path.isfile(conflict_csv_path)
                                with open(conflict_csv_path, 'a', newline='', encoding='utf-8') as cf:
                                    w = csv.writer(cf)
                                    if write_header:
                                        w.writerow(['timestamp','epoch','trace_id','is_host_fault','groundtruth','gt_int','gt_rank','gt_score','top1_kind','top1_id','top1_score','delta'])
                                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    w.writerow([ts, (int(epoch) if epoch is not None else ''), str(trace_id), False, gt, int(gt_int), int(gt_rank), float(gt_score), str(k0), int(i0), float(s0), float(delta)])
                        except Exception:
                            pass

        # Score scale diagnostics CSV（便于量纲对比）
        try:
            _svc_max_for_debug = max(svc_final.values()) if svc_final else 0.0
            _host_max_for_debug = max(host_final.values()) if host_final else 0.0
            dbg_dir = os.path.dirname(conflict_csv_path) if conflict_csv_path else None
            dbg_path = os.path.join(dbg_dir, 'score_distribution_debug.csv') if dbg_dir else 'score_distribution_debug.csv'
            write_header = not os.path.isfile(dbg_path)
            with open(dbg_path, 'a', newline='', encoding='utf-8') as dfp:
                w = csv.writer(dfp)
                if write_header:
                    w.writerow(['timestamp','epoch','trace_id','gt_type','svc_max','host_max','lambda_host'])
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                gt_type = 'host' if is_host_fault_k else 'svc'
                w.writerow([ts, (int(epoch) if epoch is not None else ''), str(trace_id), gt_type, float(_svc_max_for_debug), float(_host_max_for_debug), float(lambda_host)])
        except Exception:
            pass

        results.append(rec)

    acc_top1 = (top1 / total) if total > 0 else 0.0
    acc_topk = (topk_correct / total) if total > 0 else 0.0
    acc_svc_top1 = (svc_top1 / svc_total) if svc_total > 0 else 0.0
    acc_svc_topk = (svc_topk / svc_total) if svc_total > 0 else 0.0
    acc_host_top1 = (host_top1 / host_total) if host_total > 0 else 0.0
    acc_host_topk = (host_topk / host_total) if host_total > 0 else 0.0
    return results, acc_top1, acc_topk, acc_svc_top1, acc_svc_topk, acc_host_top1, acc_host_topk


__all__ = ['evaluate_mixed_root_cause_urs', 'evaluate_mixed_root_cause_rerank']