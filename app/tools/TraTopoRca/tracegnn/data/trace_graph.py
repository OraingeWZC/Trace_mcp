import os
import pickle as pkl
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import *

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import *

__all__ = [
    'TraceGraphNodeFeatures',
    'TraceGraphNodeReconsScores',
    'TraceGraphNode',
    'TraceGraphVectors',
    'TraceGraph',
    'TraceGraphIDManager',
    'load_trace_csv',
    'df_to_trace_graphs',
]


SERVICE_ID_YAML_FILE = 'service_id.yml'
OPERATION_ID_YAML_FILE = 'operation_id.yml'
STATUS_ID_YAML_FILE = 'status_id.yml'
FAULT_CATEGORY_YAML_FILE = 'fault_category.yml'
HOST_ID_YAML_FILE = 'host_id.yml'


@dataclass
class TraceGraphNodeFeatures(object):
    __slots__ = ['span_count', 'max_latency', 'min_latency', 'avg_latency']

    span_count: int  # number of duplicates in the parent
    avg_latency: float  # for span_count == 1, avg == max == min
    max_latency: float
    min_latency: float


@dataclass
class TraceGraphNodeReconsScores(object):
    # probability of the node
    edge_logit: float
    operation_logit: float

    # probability of the latency
    avg_latency_nstd: float  # (avg_latency - avg_latency_mean) / avg_latency_std


@dataclass
class TraceGraphSpan(object):
    __slots__ = [
        'span_id', 'start_time', 'latency', 'status'
    ]

    span_id: Optional[int]
    start_time: Optional[datetime]
    latency: float
    status: str


@dataclass
class TraceGraphNode(object):
    __slots__ = [
        'node_id', 'service_id', 'operation_id', 'status_id', 'host_id',
        'features', 'children', 'spans', 'scores'
    ]

    node_id: Optional[int]  # the node id of the graph
    service_id: Optional[int]  # the service id
    status_id: Optional[int]
    host_id: Optional[int]  # the host id (NodeName)
    operation_id: int  # the operation id
    features: TraceGraphNodeFeatures  # the node features
    children: List['TraceGraphNode']  # children nodes
    spans: Optional[List[TraceGraphSpan]]  # detailed spans information (from the original data)
    scores: Optional[TraceGraphNodeReconsScores]

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return id(self)

    @staticmethod
    def new_sampled(node_id: int,
                    operation_id: int,
                    status_id: int,
                    features: TraceGraphNodeFeatures,
                    scores: Optional[TraceGraphNodeReconsScores] = None
                    ):
        return TraceGraphNode(
            node_id=node_id,
            service_id=None,
            operation_id=operation_id,
            status_id=status_id,
            features=features,
            children=[],
            spans=None,
            scores=scores
        )

    def iter_bfs(self,
                 depth: int = 0,
                 with_parent: bool = False
                 ) -> Generator[
                    Union[
                        Tuple[int, 'TraceGraphNode'],
                        Tuple[int, int, 'TraceGraphNode', 'TraceGraphNode']
                    ],
                    None,
                    None
                ]:
        """Iterate through the nodes in BFS order."""
        if with_parent:
            depth = depth
            level = [(self, None, 0)]

            while level:
                next_level: List[Tuple[TraceGraphNode, TraceGraphNode, int]] = []
                for nd, parent, idx in level:
                    yield depth, idx, nd, parent
                    for c_idx, child in enumerate(nd.children):
                        next_level.append((child, nd, c_idx))
                depth += 1
                level = next_level

        else:
            depth = depth
            level = [self]

            while level:
                next_level: List[TraceGraphNode] = []
                for nd in level:
                    yield depth, nd
                    next_level.extend(nd.children)
                depth += 1
                level = next_level

    def count_nodes(self) -> int:
        ret = 0
        for _ in self.iter_bfs():
            ret += 1
        return ret


@dataclass
class TraceGraphVectors(object):
    """Cached result of `TraceGraph.graph_vectors()`."""
    __slots__ = [
        'u', 'v',
        'node_type',
        'node_depth', 'node_idx',
        'span_count', 'avg_latency', 'max_latency', 'min_latency',
        'node_features', 'status'
    ]

    # note that it is guaranteed that u[i] < v[i], i.e., upper triangle matrix
    u: np.ndarray
    v: np.ndarray

    # node type
    node_type: np.ndarray

    # node depth
    node_depth: np.ndarray

    # node idx
    node_idx: np.ndarray

    # node feature
    span_count: np.ndarray
    avg_latency: np.ndarray
    max_latency: np.ndarray
    min_latency: np.ndarray

    # status
    status: List[str]


@dataclass
class TraceGraph(object):
    __slots__ = [
        'version',
        'trace_id', 'parent_id', 'root', 'node_count', 'max_depth', 'data', 'status', 'anomaly', 'root_cause', 'fault_category'
    ]

    version: int  # version control
    trace_id: Optional[Tuple[int, int]]
    parent_id: Optional[int]
    root: TraceGraphNode
    node_count: Optional[int]
    max_depth: Optional[int]
    anomaly: int  # 0 normal, 1 abnormal
    root_cause: Optional[int]  # root cause of the anomaly
    fault_category: Optional[int]  # fault category of the anomaly
    data: Dict[str, Any]  # any data about the graph
    status: Set[str]

    @staticmethod
    def default_version() -> int:
        return 0x2

    @staticmethod
    def new_sampled(root: TraceGraphNode, node_count: int, max_depth: int):
        return TraceGraph(
            version=TraceGraph.default_version(),
            trace_id=None,
            parent_id=None,
            root=root,
            node_count=node_count,
            max_depth=max_depth,
            data={},
            anomaly=0,
            root_cause=None,
            fault_category=None,
            status=set()
        )

    @property
    def edge_count(self) -> Optional[int]:
        if self.node_count is not None:
            return self.node_count - 1

    def iter_bfs(self,
                 with_parent: bool = False
                 ):
        """Iterate through the nodes in BFS order."""
        yield from self.root.iter_bfs(with_parent=with_parent)

    def merge_spans_and_assign_id(self):
        """
        Merge spans with the same (service, operation) under the same parent,
        and re-assign node IDs.
        """
        node_count = 0
        max_depth = 0

        for depth, parent in self.iter_bfs():
            max_depth = max(max_depth, depth)

            # assign ID to this node
            parent.node_id = node_count
            node_count += 1

            # merge the children of this node
            children = []
            for child in sorted(parent.children, key=lambda o: o.operation_id):
                if children and children[-1].operation_id == child.operation_id:
                    prev_child = children[-1]

                    # merge the features
                    f1, f2 = prev_child.features, child.features
                    f1.span_count += f2.span_count
                    f1.avg_latency += (f2.avg_latency - f1.avg_latency) * (f2.span_count / f1.span_count)
                    f1.max_latency = max(f1.max_latency, f2.max_latency)
                    f1.min_latency = min(f1.min_latency, f2.min_latency)

                    # merge the children
                    if child.children:
                        if prev_child.children:
                            prev_child.children.extend(child.children)
                        else:
                            prev_child.children = child.children

                    # merge the spans
                    if child.spans:
                        if prev_child.spans:
                            prev_child.spans.extend(child.spans)
                        else:
                            prev_child.spans = child.spans
                else:
                    children.append(child)

            # re-assign the merged children
            parent.children = children

        # record node count and depth
        self.node_count = node_count
        self.max_depth = max_depth

    def assign_node_id(self):
        """Assign node IDs to the graph nodes by pre-root order."""
        node_count = 0
        max_depth = 0

        for depth, node in self.iter_bfs():
            max_depth = max(max_depth, depth)

            # assign id to this node
            node.node_id = node_count
            node_count += 1

        # record node count and depth
        self.node_count = node_count
        self.max_depth = max_depth

    def graph_vectors(self):
        # edge index
        u = np.empty([self.edge_count], dtype=np.int64)
        v = np.empty([self.edge_count], dtype=np.int64)

        # node type
        node_type = np.zeros([self.node_count], dtype=np.int64)

        # node depth
        node_depth = np.zeros([self.node_count], dtype=np.int64)

        # node idx
        node_idx = np.zeros([self.node_count], dtype=np.int64)

        # node feature
        span_count = np.zeros([self.node_count], dtype=np.int64)
        avg_latency = np.zeros([self.node_count], dtype=np.float32)
        max_latency = np.zeros([self.node_count], dtype=np.float32)
        min_latency = np.zeros([self.node_count], dtype=np.float32)
        
        # status
        status = [''] * self.node_count

        # X = np.zeros([self.node_count, x_dim], dtype=np.float32)

        edge_idx = 0
        for depth, idx, node, parent in self.iter_bfs(with_parent=True):
            j = node.node_id
            feat = node.features

            # node type
            node_type[j] = node.operation_id

            # node depth
            node_depth[j] = depth

            # node idx
            node_idx[j] = idx

            # node feature
            span_count[j] = feat.span_count
            avg_latency[j] = feat.avg_latency
            max_latency[j] = feat.max_latency
            min_latency[j] = feat.min_latency
            # X[parent.node_id, parent.operation_id] = 1   # one-hot encoded node feature
            status[j] = node.spans[0].status

            # edge index
            for child in node.children:
                u[edge_idx] = node.node_id
                v[edge_idx] = child.node_id
                edge_idx += 1

        if len(u) != self.edge_count:
            raise ValueError(f'`len(u)` != `self.edge_count`: {len(u)} != {self.edge_count}')

        return TraceGraphVectors(
            # edge index
            u=u, v=v,
            # node type
            node_type=node_type,
            # node depth
            node_depth=node_depth,
            # node idx
            node_idx=node_idx,
            # node feature
            span_count=span_count,
            avg_latency=avg_latency,
            max_latency=max_latency,
            min_latency=min_latency,
            status=status
        )

    def networkx_graph(self, id_manager: 'TraceGraphIDManager') -> nx.Graph:
        gv = self.graph_vectors()
        self_nodes = {nd.node_id: nd for _, nd in self.iter_bfs()}
        g = nx.Graph()
        # graph
        for k, v in self.data.items():
            g.graph[k] = v
        # nodes
        g.add_nodes_from(range(self.node_count))
        # edges
        g.add_edges_from([(i, j) for i, j in zip(gv.u, gv.v)])
        # node features
        for i in range(len(gv.node_type)):
            nd = g.nodes[i]
            nd['anomaly'] = self_nodes[i].anomaly
            nd['status'] = gv.status[i]
            nd['node_type'] = gv.node_type[i]
            nd['service_id'] = self_nodes[i].service_id
            nd['operation'] = id_manager.operation_id.rev(gv.node_type[i])
            for attr in TraceGraphNodeFeatures.__slots__:
                nd[attr] = getattr(gv, attr)[i]
            if self_nodes[i].scores:
                nd['avg_latency_nstd'] = self_nodes[i].scores.avg_latency_nstd
        return g

    def to_bytes(self, protocol: int = pkl.DEFAULT_PROTOCOL) -> bytes:
        return pkl.dumps(self, protocol=protocol)

    @staticmethod
    def from_bytes(content: bytes) -> 'TraceGraph':
        r = pkl.loads(content)
        return r

    def deepcopy(self) -> 'TraceGraph':
        return TraceGraph.from_bytes(self.to_bytes())


@dataclass
class TempGraphNode(object):
    __slots__ = ['trace_id', 'parent_id', 'node']

    trace_id: Tuple[int, int]
    parent_id: int
    node: 'TraceGraphNode'


class TraceGraphIDManager(object):
    __slots__ = ['root_dir', 'service_id', 'operation_id', 'status_id', 'fault_category', 'host_id']

    root_dir: str
    service_id: IDAssign
    operation_id: IDAssign
    status_id: IDAssign
    fault_category: IDAssign

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.service_id = IDAssign(os.path.join(self.root_dir, SERVICE_ID_YAML_FILE))
        self.operation_id = IDAssign(os.path.join(self.root_dir, OPERATION_ID_YAML_FILE))
        self.status_id = IDAssign(os.path.join(self.root_dir, STATUS_ID_YAML_FILE))
        self.fault_category = IDAssign(os.path.join(self.root_dir, FAULT_CATEGORY_YAML_FILE))
        self.host_id = IDAssign(os.path.join(self.root_dir, HOST_ID_YAML_FILE))

    def __enter__(self):
        self.service_id.__enter__()
        self.operation_id.__enter__()
        self.status_id.__enter__()
        self.fault_category.__enter__()
        self.host_id.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.service_id.__exit__(exc_type, exc_val, exc_tb)
        self.operation_id.__exit__(exc_type, exc_val, exc_tb)
        self.status_id.__exit__(exc_type, exc_val, exc_tb)
        self.fault_category.__exit__(exc_type, exc_val, exc_tb)
        self.host_id.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_operations(self) -> int:
        return len(self.operation_id)

    @property
    def num_services(self) -> int:
        return len(self.service_id)

    @property
    def num_status(self) -> int:
        return len(self.status_id)

    @property
    def num_fault_categories(self) -> int:
        return len(self.fault_category)

    def dump_to(self, output_dir: str):
        self.service_id.dump_to(os.path.join(output_dir, SERVICE_ID_YAML_FILE))
        self.operation_id.dump_to(os.path.join(output_dir, OPERATION_ID_YAML_FILE))
        self.status_id.dump_to(os.path.join(output_dir, STATUS_ID_YAML_FILE))
        self.fault_category.dump_to(os.path.join(output_dir, FAULT_CATEGORY_YAML_FILE))
        self.host_id.dump_to(os.path.join(output_dir, HOST_ID_YAML_FILE))


def load_trace_csv(input_path: str) -> pd.DataFrame:
    dtype = {
        'TraceID': str,
        'SpanID': str,
        'ParentID': str,
        'OperationName': str,
        'ServiceName': str,
        'StartTime': int,
        'Duration': float,
        'StatusCode': str,
        'Anomaly': bool,
        'RootCause': str,
        'FaultCategory': str,
        # 可选列：NodeName（主机名/节点名）
        'NodeName': str,
    }
    # 仅加载存在于文件中的列，避免强制要求 NodeName 必须存在
    return pd.read_csv(
        input_path,
        engine='c',
        usecols=lambda col: col in dtype,
        dtype={k: v for k, v in dtype.items() if k != 'NodeName'}  # NodeName 作为可选列保留，但不强制 dtype
    )


def df_to_trace_graphs(df: pd.DataFrame,
                       id_manager: TraceGraphIDManager,
                       min_node_count: int = 2,
                       max_node_count: int = 100,
                       summary_file: Optional[str] = None,
                       merge_spans: bool = False,
                       ) -> List[TraceGraph]:
    summary = []
    trace_spans = {}
    trace_info = {}  # 存储trace级别的信息

    # print(f"df shape: {df.shape}")
    # print(f"df columns: {df.columns}")

    # read the spans
    with id_manager:
        for i, row in enumerate(tqdm(df.itertuples(), desc='Build nodes', total=len(df))):
            # if i < 5:
            #     print(f"Row {i}: {row}")

            trace_id = row.TraceID
            span_id = row.SpanID
            parent_span_id = row.ParentID
            service_name = row.ServiceName
            operation_name = row.OperationName
            status_code = row.StatusCode

            # 打印关键字段
            # if i < 5:
            #     print(f"trace_id={trace_id}, span_id={span_id}, parent_span_id={parent_span_id}, service_name={service_name}, operation_name={operation_name}, status_code={status_code}")

            # 获取或创建该trace_id对应的span字典
            span_dict = trace_spans.get(trace_id, None)
            if span_dict is None:
                trace_spans[trace_id] = span_dict = {}
                trace_info[trace_id] = {
                    'anomaly': getattr(row, 'Anomaly', False),
                    'root_cause': getattr(row, 'RootCause', None),
                    'fault_category': getattr(row, 'FaultCategory', None),
                }
            else:
                # Some datasets only populate trace-level labels on a subset of spans.
                # If we already have the trace registered but label fields are still empty,
                # try to fill them from later rows.
                try:
                    info = trace_info.get(trace_id, {})
                    if info is not None:
                        rc_prev = info.get('root_cause', None)
                        fc_prev = info.get('fault_category', None)
                        rc_cur = getattr(row, 'RootCause', None)
                        fc_cur = getattr(row, 'FaultCategory', None)
                        if (rc_prev is None or str(rc_prev).strip() in ('', 'nan', 'None', 'null')) and rc_cur is not None:
                            info['root_cause'] = rc_cur
                        if (fc_prev is None or str(fc_prev).strip() in ('', 'nan', 'None', 'null')) and fc_cur is not None:
                            info['fault_category'] = fc_cur
                except Exception:
                    pass

            # 创建临时图节点
            span_dict[span_id] = TempGraphNode(
                trace_id=trace_id,
                parent_id=parent_span_id,
                node=TraceGraphNode(
                    node_id=None,
                    service_id=id_manager.service_id.get_or_assign(service_name),
                    operation_id=id_manager.operation_id.get_or_assign(f'{service_name}/{operation_name}'),
                    status_id=id_manager.status_id.get_or_assign(str(status_code)),
                    host_id=id_manager.host_id.get_or_assign(getattr(row, 'NodeName', '')),
                    features=TraceGraphNodeFeatures(
                        span_count=1,
                        avg_latency=row.Duration,
                        max_latency=row.Duration,
                        min_latency=row.Duration,
                    ),
                    children=[],
                    spans=[
                        TraceGraphSpan(
                            span_id=span_id,
                            start_time=row.StartTime,
                            latency=row.Duration,
                            status=str(status_code)
                        ),
                    ],
                    scores=None
                )
            )

    # print(f"trace_spans keys: {list(trace_spans.keys())[:5]}")
    # print(f"trace_info keys: {list(trace_info.keys())[:5]}")
    # print(f"Total traces: {len(trace_spans)}")

    # construct the traces
    trace_graphs: List[TraceGraph] = []

    for trace_id, trace in tqdm(trace_spans.items(), total=len(trace_spans), desc='Build graphs'):
        nodes: List[TempGraphNode] = sorted(
            trace.values(),
            key=(lambda nd: (nd.node.service_id, nd.node.operation_id, nd.node.spans[0].start_time))
        )
        status = set()
        
        info = trace_info[trace_id]
        # print(f"Building graph for trace_id={trace_id}, nodes={len(nodes)}")

        root_count = 0
        for nd in nodes:
            parent_id = nd.parent_id
            # 判断是否为根节点
            if (parent_id == '-1') or (parent_id not in trace):
                root_count += 1
                # print(f"Found root node: span_id={nd.node.node_id}, parent_id={parent_id}")
                # 将 root_cause 与 fault_category 安静转换为整型ID，无法转换则置 0
                def _to_int_or_zero(v):
                    try:
                        if isinstance(v, (int, float)):
                            return int(v)
                        s = str(v) if v is not None else ''
                        return int(float(s)) if s.replace('.', '', 1).isdigit() else 0
                    except Exception:
                        return 0

                def _to_fault_category_id(v):
                    """
                    FaultCategory may be:
                    - an int id (already mapped), or
                    - a string name like 'cpu', 'memory', 'node cpuchaos', etc.
                    Map names via id_manager.fault_category to keep evaluation consistent.
                    """
                    try:
                        if isinstance(v, (int, float)):
                            return int(v)
                        s = (str(v) if v is not None else '').strip()
                        if not s or s.lower() in ('nan', 'none', 'null'):
                            return 0
                        # numeric string -> int
                        if s.replace('.', '', 1).isdigit():
                            return int(float(s))
                        # name string -> id
                        return int(id_manager.fault_category.get_or_assign(s.lower()))
                    except Exception:
                        return 0

                root_cause_id = _to_int_or_zero(info['root_cause'])
                fault_category_id = _to_fault_category_id(info['fault_category'])
                
                trace_graphs.append(TraceGraph(
                    version=TraceGraph.default_version(),
                    trace_id=nd.trace_id,
                    parent_id=nd.parent_id,
                    root=nd.node,
                    node_count=None,
                    max_depth=None,
                    data={},
                    status=set([span.status for span in nd.node.spans]),
                    anomaly=1 if info['anomaly'] else 0,
                    root_cause=root_cause_id,
                    fault_category=fault_category_id
                ))
            else:
                trace[parent_id].node.children.append(nd.node)
                status = status.union([span.status for span in nd.node.spans])

        # print(f"Trace {trace_id} root_count={root_count}, total nodes={len(nodes)}")
        if len(nodes) < min_node_count or len(nodes) > max_node_count:
            # print(f"Trace {trace_id} filtered by node count: {len(nodes)}")
            if trace_graphs:
                trace_graphs.pop()  # 移除刚刚添加的图

        if trace_graphs:
            trace_graphs[-1].status = trace_graphs[-1].status.union(status)

    # print(f"Total trace_graphs before assign id: {len(trace_graphs)}")

    # merge spans and assign id
    if merge_spans:
        for trace in tqdm(trace_graphs, desc='Merge spans and assign node id'):
            trace.merge_spans_and_assign_id()
    else:
        for trace in tqdm(trace_graphs, desc='Assign node id'):
            trace.assign_node_id()

    # print(f"Total trace_graphs after assign id: {len(trace_graphs)}")
    return trace_graphs
