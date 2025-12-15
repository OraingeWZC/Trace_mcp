from typing import Dict, Any, Optional
import os
import numpy as np
import pandas as pd

# Infra CSV loading (merged_all_infra.csv)
INFRA_FILE_NAME = 'merged_all_infra.csv'
INFRA_METRICS = [
    'node_cpu_usage_rate',
    'node_memory_usage_rate',
    'node_filesystem_usage_rate',
    'node_disk_read_time_seconds_total',
    'node_disk_write_time_seconds_total',
]


def _find_infra_path(processed_dir: str) -> Optional[str]:
    """Locate merged_all_infra.csv near the processed dir.
    Priority:
      1) <processed_dir>/merged_all_infra.csv
      2) <processed_dir>/infra/merged_all_infra.csv
      3) <dataset_root>/infra/merged_all_infra.csv (one-level up if processed)
    """
    p0 = os.path.join(processed_dir, INFRA_FILE_NAME)
    if os.path.isfile(p0):
        return p0
    p1 = os.path.join(processed_dir, 'infra', INFRA_FILE_NAME)
    if os.path.isfile(p1):
        return p1
    root = os.path.dirname(processed_dir.rstrip(os.sep))
    p2 = os.path.join(root, 'infra', INFRA_FILE_NAME)
    if os.path.isfile(p2):
        return p2
    return None


def load_host_infra_index(processed_dir: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load infra CSV and build an index: host_name -> { 'timeMs': ndarray[int64], 'metrics': {metric: ndarray[float64]} }.
    Returns None if file not found or failed to parse.
    """
    path = _find_infra_path(processed_dir)
    if not path or not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # Ensure required columns exist
    if 'timeMs' not in df.columns or 'kubernetes_node' not in df.columns:
        return None
    # Add missing metrics with NaN if absent
    for m in INFRA_METRICS:
        if m not in df.columns:
            df[m] = np.nan
    df = df[['timeMs', 'kubernetes_node'] + INFRA_METRICS].copy()
    df = df.dropna(subset=['timeMs', 'kubernetes_node'])
    # Normalize dtypes
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

    host_idx: Dict[str, Dict[str, Any]] = {}
    for host, g in df.groupby('kubernetes_node'):
        lg = g.sort_values('timeMs')
        host_idx[str(host)] = {
            'timeMs': lg['timeMs'].to_numpy(dtype=np.int64),
            'metrics': {m: lg[m].to_numpy(dtype=np.float64) for m in INFRA_METRICS}
        }
    return host_idx


__all__ = [
    'INFRA_FILE_NAME',
    'INFRA_METRICS',
    '_find_infra_path',
    'load_host_infra_index',
]

