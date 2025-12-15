from typing import Dict, Set, Any, Optional
import os
import yaml
import numpy as np

from tracegnn.data.trace_graph import TraceGraphIDManager


def load_host_topology(processed_dir: str,
                       id_manager: TraceGraphIDManager
                       ) -> Dict[int, Set[int]]:
    path = os.path.join(processed_dir, 'host_topology.yml')
    adj: Dict[int, Set[int]] = {}
    if not os.path.isfile(path):
        return adj

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception:
        return adj

    def to_host_id(x: Any) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x).strip()
        if s == '':
            return None
        try:
            return int(s)
        except Exception:
            hid = id_manager.host_id.get(s)
            return int(hid) if hid is not None else None

    def add_undirected(u: Optional[int], v: Optional[int]):
        if u is None or v is None or u == v:
            return
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    if isinstance(data, list):
        for e in data:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                add_undirected(to_host_id(e[0]), to_host_id(e[1]))
    elif isinstance(data, dict):
        for k, vs in data.items():
            u = to_host_id(k)
            if not isinstance(vs, (list, tuple)):
                continue
            for v in vs:
                add_undirected(u, to_host_id(v))

    return adj


def _propagate_on_hosts(host_scores: Dict[int, float],
                        host_adj: Dict[int, Set[int]],
                        alpha: float = 0.7,
                        steps: int = 1) -> Dict[int, float]:
    if not host_scores:
        return {}
    s = dict(host_scores)
    for _ in range(max(0, int(steps))):
        new_s: Dict[int, float] = {}
        for h, v in s.items():
            neighs = list(host_adj.get(int(h), []))
            if neighs:
                m = float(sum(s.get(int(n), 0.0) for n in neighs)) / max(len(neighs), 1)
                new_s[h] = float(v) + float(alpha) * m
            else:
                new_s[h] = float(v)
        s = new_s
    return s


__all__ = ['load_host_topology', '_propagate_on_hosts']

