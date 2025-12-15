from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np

from tracegnn.data.trace_graph import TraceGraphIDManager
from .sinfra_core import (
    _minute_key_from_graph,
    _robust_normalize,
    compute_sinfra_per_host,
    compute_sinfra_per_host_with_components,
)
from .topology import _propagate_on_hosts


def _host_pool_scores(node_scores: np.ndarray,
                      host_ids: np.ndarray,
                      topk_pool: int = 3) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = {}
    for i in range(len(node_scores)):
        h = int(host_ids[i])
        if h <= 0:
            continue
        buckets.setdefault(h, []).append(float(node_scores[i]))
    pooled: Dict[int, float] = {}
    for h, scores in buckets.items():
        if not scores:
            continue
        scores.sort(reverse=True)
        pooled[h] = float(sum(scores[:max(1, int(topk_pool))]))
    return pooled


def evaluate_with_host_topology_infra(
    all_trace_info: List[Dict[str, Any]],
    true_root_causes: Dict[str, Any],
    host_adj: Dict[int, Set[int]],
    topk: int = 5,
    pool_k: int = 3,
    alpha: float = 0.7,
    steps: int = 1,
    fault_category_names: Optional[Dict[int, str]] = None,
    id_manager: Optional[TraceGraphIDManager] = None,
    infra_index: Optional[Dict[str, Dict[str, Any]]] = None,
    W: int = 3,
    eta: float = 0.3,
    host_nll_key: Optional[str] = None,
    lambda_host: float = 0.5,
    rho_mode: str = 'count',
    debug: bool = False,
    ms_W_list: Optional[List[int]] = None,
    lse_tau: float = 2.0,
    sinfra_w: Optional[Dict[str, float]] = None,
    peer_mode: str = 'trace',
) -> Tuple[List[Dict[str, Any]], float, float]:
    results: List[Dict[str, Any]] = []
    total = top1 = topk_correct = 0

    host_name: Dict[int, str] = {}
    if id_manager is not None:
        try:
            host_name = {i: id_manager.host_id.rev(i) for i in range(len(id_manager.host_id))}
        except Exception:
            host_name = {}

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
        node_scores = trace['node_scores']
        host_ids = g.ndata['host_id']
        fc_id = trace.get('fault_category', None)

        is_host_fault = False
        if fc_id is not None and fault_category_names:
            name = fault_category_names.get(int(fc_id), '').lower()
            is_host_fault = name.startswith('node') or ('host' in name)
        else:
            try:
                is_host_fault = gt_int in set(map(int, host_ids.detach().cpu().numpy().tolist()))
            except Exception:
                is_host_fault = False
        if not is_host_fault:
            continue

        ns = node_scores.detach().cpu().numpy()
        hids = host_ids.detach().cpu().numpy()
        pooled_model = _host_pool_scores(ns, hids, topk_pool=pool_k)

        uniq_hosts = sorted({int(h) for h in hids if int(h) > 0})
        rho: Dict[int, float] = {}
        if uniq_hosts:
            if rho_mode == 'duration' and 'latency' in g.ndata and 'span_count' in g.ndata:
                lat = g.ndata['latency'].detach().cpu().numpy()
                spc = g.ndata['span_count'].detach().cpu().numpy()
                val: Dict[int, float] = {}
                for i in range(len(hids)):
                    hid = int(hids[i]);
                    if hid <= 0: continue
                    val[hid] = val.get(hid, 0.0) + float(lat[i]) * float(spc[i])
            else:
                spc = g.ndata['span_count'].detach().cpu().numpy() if 'span_count' in g.ndata else np.ones_like(hids)
                val = {}
                for i in range(len(hids)):
                    hid = int(hids[i]);
                    if hid <= 0: continue
                    val[hid] = val.get(hid, 0.0) + float(spc[i])
            v = np.array([val.get(h, 0.0) for h in uniq_hosts], dtype=np.float64)
            if np.all(v == 0): v = np.ones_like(v)
            v = np.exp(v - v.max()); v = v / max(v.sum(), 1e-6)
            rho = {h: float(v[i]) for i, h in enumerate(uniq_hosts)}

        t0_min_ms = _minute_key_from_graph(g)
        sinfra: Dict[int, float] = {}
        sinfra_comp: Dict[int, Dict[str, float]] = {}
        if infra_index is not None and t0_min_ms is not None and id_manager is not None:
            peer_hosts_names: List[str] = []
            try:
                peer_hosts_names = [host_name.get(h, None) for h in uniq_hosts if host_name.get(h, None)]
            except Exception:
                peer_hosts_names = []
            for hid in uniq_hosts:
                hnm = host_name.get(hid, None)
                if not hnm: continue
                if debug:
                    v, comp = compute_sinfra_per_host_with_components(
                        hnm, infra_index, t0_min_ms, W=W,
                        ms_W_list=ms_W_list, lse_tau=lse_tau, sinfra_w=sinfra_w,
                        peer_mode=peer_mode, peer_hosts=peer_hosts_names,
                    )
                    if v is not None:
                        sinfra[hid] = float(v)
                        sinfra_comp[hid] = comp
                else:
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

        host_score: Dict[int, float] = {}
        all_h = set(sinfra_n.keys()) | set(pooled_model.keys())
        for h in all_h:
            host_score[h] = lambda_host * float(sinfra_n.get(h, 0.0)) + (1.0 - lambda_host) * float(pooled_model.get(h, 0.0))

        host_score_pre = dict(host_score)
        host_score = _propagate_on_hosts(host_score, host_adj, alpha=alpha, steps=steps)
        ranked = sorted(host_score.items(), key=lambda kv: kv[1], reverse=True)
        top_host_ids = [h for h, s in ranked[:topk]]

        rec = {
            'trace_id': trace_id,
            'groundtruth': gt,
            'fault_category': fc_id,
            'top_host': ranked[:topk],
            'is_correct_host_top1': False,
            'is_correct_host_topk': False,
        }

        if debug:
            hit_hosts = len(sinfra)
            total_hosts = len(uniq_hosts)
            hit_ratio = (hit_hosts / max(total_hosts, 1)) if total_hosts > 0 else 0.0
            if host_score_pre:
                top1_pre = max(host_score_pre.items(), key=lambda kv: kv[1])[0]
                rec.update({
                    'infra_total_hosts': total_hosts,
                    'infra_hit_hosts': hit_hosts,
                    'infra_hit_ratio': hit_ratio,
                    'minute_key_ms': int(t0_min_ms) if t0_min_ms is not None else None,
                    'W': int(W),
                    'rho_mode': str(rho_mode),
                    'lambda_host': float(lambda_host),
                    'eta': float(eta),
                    'alpha': float(alpha),
                    'top1_pre_host': int(top1_pre),
                    'top1_pre_model_n': float(_robust_normalize(pooled_model).get(top1_pre, 0.0)),
                    'top1_pre_infra_n': float(sinfra_n.get(top1_pre, 0.0)),
                    'top1_pre_rho': float(rho.get(top1_pre, 0.0)),
                    'top1_pre_fused': float(host_score_pre.get(top1_pre, 0.0)),
                    'top1_pre_sinfra_components': ';'.join([f"{k}:{v:.4f}" for k, v in sorted(sinfra_comp.get(top1_pre, {}).items())]) if sinfra_comp else '',
                })

        total += 1
        if top_host_ids:
            if int(top_host_ids[0]) == gt_int:
                top1 += 1; rec['is_correct_host_top1'] = True
            if gt_int in list(map(int, top_host_ids)):
                topk_correct += 1; rec['is_correct_host_topk'] = True

        results.append(rec)

    acc_top1 = (top1 / total) if total > 0 else 0.0
    acc_topk = (topk_correct / total) if total > 0 else 0.0
    return results, acc_top1, acc_topk


__all__ = ['evaluate_with_host_topology_infra']

