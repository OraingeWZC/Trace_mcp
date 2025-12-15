from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .infra_io import INFRA_METRICS


def _minute_key_from_graph(g) -> Optional[int]:
    try:
        st = g.ndata.get('start_time', None)
        if st is None or len(st) == 0:
            return None
        t0_s = int(st.min().item())
        t0_ms = t0_s * 1000
        return (t0_ms // 60000) * 60000
    except Exception:
        return None


def _window_indices(t_ms: np.ndarray, t0_min_ms: int, W: int) -> np.ndarray:
    lo = t0_min_ms - W * 60000
    hi = t0_min_ms
    return np.where((t_ms >= lo) & (t_ms <= hi))[0]


def _robust_s_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if x.size == 0:
        return x
    med = np.median(x)
    q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = max(q3 - q1, eps)
    return np.clip(np.abs(x - med) / iqr, 0.0, 6.0)


def _logsumexp(vals: List[float], tau: float = 1.0) -> float:
    if not vals:
        return 0.0
    v = np.asarray(vals, dtype=np.float64)
    m = float(np.max(v))
    return float(tau * (np.log(np.sum(np.exp((v - m) / max(tau, 1e-6)))) + (m / max(tau, 1e-6))))


def _nearest_leq_idx(t_ms: np.ndarray, t0_min_ms: int) -> Optional[int]:
    if t_ms.size == 0:
        return None
    pos = int(np.searchsorted(t_ms, t0_min_ms, side='right')) - 1
    return pos if (pos >= 0) else None


def _robust_normalize(score_dict: Dict[int, float], eps: float = 1e-6) -> Dict[int, float]:
    if not score_dict:
        return {}
    vals = np.array(list(score_dict.values()), dtype=float)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad < eps:
        std = float(np.std(vals))
        denom = std if std > eps else 1.0
    else:
        denom = mad
    return {k: (float(v) - med) / denom for k, v in score_dict.items()}


def compute_sinfra_per_host(
    host_name: str,
    infra_index: Optional[Dict[str, Dict[str, Any]]],
    t0_min_ms: Optional[int],
    W: int = 3,
    metric_weights: Optional[Dict[str, float]] = None,
    ms_W_list: Optional[List[int]] = None,
    lse_tau: float = 2.0,
    sinfra_w: Optional[Dict[str, float]] = None,
    peer_mode: str = 'trace',
    peer_hosts: Optional[List[str]] = None,
) -> Optional[float]:
    if infra_index is None or t0_min_ms is None:
        return None
    h = infra_index.get(str(host_name))
    if not h:
        return None

    weights = metric_weights or {m: 1.0 for m in INFRA_METRICS}
    if not ms_W_list:
        t = h['timeMs']
        sel = _window_indices(t, t0_min_ms, W)
        if sel.size == 0:
            return None
        total = 0.0
        for m in INFRA_METRICS:
            arr = h['metrics'][m][sel]
            if arr.size == 0 or np.all(np.isnan(arr)):
                continue
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            s = _robust_s_z(arr)
            total += float(weights.get(m, 1.0)) * float(np.max(s))
        return total

    wcfg = sinfra_w or {'z_point': 0.6, 'z_win': 0.3, 'peer': 0.1}
    z_point_total = 0.0
    z_win_total = 0.0
    peer_total = 0.0
    t = h['timeMs']
    for m in INFRA_METRICS:
        arr_all = h['metrics'].get(m, None)
        if arr_all is None:
            continue
        z_point_ws: List[float] = []
        z_win_ws: List[float] = []
        for Wm in ms_W_list:
            sel = _window_indices(t, t0_min_ms, int(Wm))
            if sel.size == 0:
                continue
            arr = arr_all[sel]
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            s = _robust_s_z(arr)
            z_point_ws.append(float(s[-1]))
            z_win_ws.append(float(np.max(s)))
        if z_point_ws:
            z_point_total += float(weights.get(m, 1.0)) * _logsumexp(z_point_ws, tau=lse_tau)
        if z_win_ws:
            z_win_total += float(weights.get(m, 1.0)) * _logsumexp(z_win_ws, tau=lse_tau)

        idx = _nearest_leq_idx(t, t0_min_ms)
        if idx is not None:
            x0 = float(arr_all[idx])
            peers: List[float] = []
            if peer_mode == 'trace' and peer_hosts:
                for ph in peer_hosts:
                    if ph == host_name:
                        continue
                    hh = infra_index.get(str(ph))
                    if not hh:
                        continue
                    ti = _nearest_leq_idx(hh['timeMs'], t0_min_ms)
                    if ti is None:
                        continue
                    val = hh['metrics'].get(m, None)
                    if val is None:
                        continue
                    v = float(val[ti])
                    if not np.isnan(v):
                        peers.append(v)
            elif peer_mode == 'global':
                for ph, hh in infra_index.items():
                    if ph == host_name:
                        continue
                    ti = _nearest_leq_idx(hh['timeMs'], t0_min_ms)
                    if ti is None:
                        continue
                    val = hh['metrics'].get(m, None)
                    if val is None:
                        continue
                    v = float(val[ti])
                    if not np.isnan(v):
                        peers.append(v)
            if len(peers) >= 3:
                med = float(np.median(peers))
                mad = float(np.median(np.abs(np.asarray(peers, dtype=np.float64) - med)))
                denom = mad if mad > 1e-6 else float(np.std(peers))
                denom = denom if denom > 1e-6 else 1.0
                peer_total += float(weights.get(m, 1.0)) * float(np.clip(abs(x0 - med) / denom, 0.0, 6.0))

    total = float(wcfg.get('z_point', 0.6)) * z_point_total \
          + float(wcfg.get('z_win', 0.3)) * z_win_total \
          + float(wcfg.get('peer', 0.1)) * peer_total
    return total


def compute_sinfra_per_host_with_components(
    host_name: str,
    infra_index: Optional[Dict[str, Dict[str, Any]]],
    t0_min_ms: Optional[int],
    W: int = 3,
    metric_weights: Optional[Dict[str, float]] = None,
    ms_W_list: Optional[List[int]] = None,
    lse_tau: float = 2.0,
    sinfra_w: Optional[Dict[str, float]] = None,
    peer_mode: str = 'trace',
    peer_hosts: Optional[List[str]] = None,
) -> Tuple[Optional[float], Dict[str, float]]:
    comp: Dict[str, float] = {}
    total = compute_sinfra_per_host(host_name, infra_index, t0_min_ms, W=W, metric_weights=metric_weights,
                                    ms_W_list=ms_W_list, lse_tau=lse_tau, sinfra_w=sinfra_w,
                                    peer_mode=peer_mode, peer_hosts=peer_hosts)
    if infra_index is None or t0_min_ms is None:
        return None, comp
    h = infra_index.get(str(host_name))
    if not h:
        return total, comp
    weights = metric_weights or {m: 1.0 for m in INFRA_METRICS}

    if not ms_W_list:
        t = h['timeMs']
        sel = _window_indices(t, t0_min_ms, W)
        if sel.size == 0:
            return total, comp
        for m in INFRA_METRICS:
            arr = h['metrics'][m][sel]
            if arr.size == 0 or np.all(np.isnan(arr)):
                continue
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            s = _robust_s_z(arr)
            comp[m] = float(np.max(s)) * float(weights.get(m, 1.0))
        return total, comp

    # multi-scale components (sparse detail for brevity)
    z_point_total = 0.0
    z_win_total = 0.0
    peer_total = 0.0
    t = h['timeMs']
    for m in INFRA_METRICS:
        arr_all = h['metrics'].get(m, None)
        if arr_all is None:
            continue
        z_point_ws: List[float] = []
        z_win_ws: List[float] = []
        for Wm in ms_W_list:
            sel = _window_indices(t, t0_min_ms, int(Wm))
            if sel.size == 0:
                continue
            arr = arr_all[sel]
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            s = _robust_s_z(arr)
            z_point_ws.append(float(s[-1]))
            z_win_ws.append(float(np.max(s)))
        if z_point_ws:
            v = _logsumexp(z_point_ws, tau=lse_tau)
            comp[f"{m}:point_ms"] = float(weights.get(m, 1.0)) * v
            z_point_total += float(weights.get(m, 1.0)) * v
        if z_win_ws:
            v = _logsumexp(z_win_ws, tau=lse_tau)
            comp[f"{m}:win_ms"] = float(weights.get(m, 1.0)) * v
            z_win_total += float(weights.get(m, 1.0)) * v
    return total, comp


__all__ = [
    '_minute_key_from_graph', '_window_indices', '_robust_s_z', '_logsumexp', '_nearest_leq_idx',
    '_robust_normalize',
    'compute_sinfra_per_host', 'compute_sinfra_per_host_with_components',
]

