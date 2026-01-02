import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd


INFRA_FILE_NAME = 'merged_all_infra.csv'

DEFAULT_METRICS = [
    'node_cpu_usage_rate',
    'node_memory_usage_rate',
    'node_filesystem_usage_rate',
]

DISK_METRICS = [
    'node_disk_read_time_seconds_total',
    'node_disk_write_time_seconds_total',
]


def _find_infra_path(processed_dir: str) -> Optional[str]:
    """
    Locate merged_all_infra.csv with recursive parent directory search.
    覆盖范围：当前目录 -> processed -> dataset_name -> dataset (TraTopoRca/dataset)
    """
    # 1. 优先检查当前传入目录及其 infra 子目录
    candidates = [
        os.path.join(processed_dir, INFRA_FILE_NAME),
        os.path.join(processed_dir, 'infra', INFRA_FILE_NAME),
    ]
    
    # 2. 向上递归查找父目录 (最多找 5 层)
    curr = processed_dir
    for _ in range(5):
        curr = os.path.dirname(curr)
        if not curr or curr == os.path.sep:
            break
        candidates.append(os.path.join(curr, INFRA_FILE_NAME))
        candidates.append(os.path.join(curr, 'infra', INFRA_FILE_NAME))

    # 3. 额外保险：检查当前运行脚本所在目录下的 dataset 文件夹
    candidates.append(os.path.join(os.getcwd(), 'dataset', INFRA_FILE_NAME))
    candidates.append(os.path.join(os.getcwd(), INFRA_FILE_NAME))

    # 执行检查并返回第一个存在的路径
    for path in candidates:
        if os.path.isfile(path):
            return path
            
    return None


def load_host_infra_index(processed_dir: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load merged infra CSV and build host-> {timeMs, metrics{m: ndarray}} index."""
    path = _find_infra_path(processed_dir)
    if not path or not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if 'timeMs' not in df.columns or 'kubernetes_node' not in df.columns:
        return None
    # normalize
    try:
        df['timeMs'] = df['timeMs'].astype(np.int64)
    except Exception:
        if 'time' in df.columns:
            try:
                df['timeMs'] = pd.to_datetime(df['time']).astype('int64') // 10**6
            except Exception:
                return None
        else:
            return None
    # keep numeric columns for metrics (present or not)
    cols = ['timeMs', 'kubernetes_node'] + [c for c in DEFAULT_METRICS + DISK_METRICS if c in df.columns]
    df = df[cols].copy()
    df = df.dropna(subset=['timeMs', 'kubernetes_node'])
    host_idx: Dict[str, Dict[str, Any]] = {}
    for host, g in df.groupby('kubernetes_node'):
        lg = g.sort_values('timeMs')
        host_idx[str(host)] = {
            'timeMs': lg['timeMs'].to_numpy(dtype=np.int64),
            'metrics': {m: lg[m].to_numpy(dtype=np.float64) for m in lg.columns if m not in ('timeMs', 'kubernetes_node')}
        }
    return host_idx


def _robust_z(arr: np.ndarray) -> np.ndarray:
    med = np.median(arr)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr <= 1e-6:
        std = np.std(arr)
        denom = std if std > 1e-6 else 1.0
    else:
        denom = iqr
    z = np.clip(np.abs(arr - med) / denom, 0.0, 6.0)
    return z


def _select_window(ts: np.ndarray, t0_min_ms: int, W: int) -> np.ndarray:
    lo = t0_min_ms - W * 60000
    hi = t0_min_ms
    return np.where((ts > lo) & (ts <= hi))[0]


def host_state_vector(
    host_name: str,
    infra_index: Optional[Dict[str, Dict[str, Any]]],
    t0_min_ms: Optional[int],
    metrics: Optional[List[str]] = None,
    W: int = 3,
    per_metric_dims: int = 3,
) -> Optional[np.ndarray]:
    """Build host_state vector for one host at one trace time window.
    Per metric we compute up to 4 dims: [z0, delta, max_z, mean_z].
    The returned per-metric slice is truncated/filled to `per_metric_dims`.
    Returns None if no infra.
    """
    if infra_index is None or t0_min_ms is None:
        return None
    h = infra_index.get(str(host_name))
    if not h:
        return None
    ts = h['timeMs']
    sel = _select_window(ts, t0_min_ms, W)
    if sel.size == 0:
        return None
    ms = metrics or DEFAULT_METRICS
    vec: List[float] = []
    for m in ms:
        arr_all = h['metrics'].get(m, None)
        if arr_all is None:
            vec.extend([0.0] * per_metric_dims)
            continue
        arr = arr_all[sel]
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            vec.extend([0.0] * per_metric_dims)
            continue
        z = _robust_z(arr)
        z0 = float(z[-1])
        if arr.size >= 2:
            med_prev = float(np.median(arr[:-1]))
            delta = float(arr[-1] - med_prev)
        else:
            delta = 0.0
        maxz = float(np.max(z))
        meanz = float(np.mean(z))
        # Assemble per-metric slice with up to 4 dims then truncate/pad
        per = [z0, delta, maxz, meanz]
        if per_metric_dims <= len(per):
            vec.extend(per[:per_metric_dims])
        else:
            # pad with zeros if a larger dim is requested
            vec.extend(per + [0.0] * (per_metric_dims - len(per)))
    return np.asarray(vec, dtype=np.float32)
