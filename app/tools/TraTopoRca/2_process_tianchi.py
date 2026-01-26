import os
import argparse
import shutil
import sqlite3
import pandas as pd
import numpy as np
import torch
import pickle  # [新增] 用于保存索引文件
from tqdm import tqdm

# 引入项目依赖
from tracegnn.data.trace_graph import df_to_trace_graphs, TraceGraphIDManager
from tracegnn.data.trace_graph_db import TraceGraphDB, BytesSqliteDB
from tracegnn.utils.host_state import host_state_vector

# ================= 配置区域 =================
DEFAULT_DATASET_ROOT = 'dataset/tianchi/2e5_1622' 

# [修改] 现在统一使用合并后的文件名
INFRA_FILENAME = 'merged_all_infra.csv'

# Host Sequence 配置 (需与 config.py 保持一致)
SEQ_WINDOW = 15
SEQ_METRICS = ['cpu', 'mem', 'disk', 'net', 'tcp'] 

# 天池数据的真实指标列名 (用于从 CSV 中提取数据)
# 脚本会去 CSV 里找这些列，如果你的合并脚本改名了，这里也要对应修改
TIANCHI_METRICS = [
    "aggregate_node_cpu_usage",
    "aggregate_node_memory_usage",
    "aggregate_node_disk_io_usage",
    "aggregate_node_net_receive_packages_errors_per_minute",
    "aggregate_node_tcp_alloc_total_num",
    "aggregate_node_tcp_inuse_total_num"
]
# ===========================================

def flexible_load_trace_csv(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(input_path)
        if 'Duration' in df.columns: df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
        if 'StartTimeMs' in df.columns: df['StartTime'] = pd.to_numeric(df['StartTimeMs'], errors='coerce')
        if 'Anomaly' in df.columns: df['Anomaly'] = df['Anomaly'].astype(bool)
        return df
    except Exception as e:
        print(f"加载CSV出错 {input_path}: {e}")
        return pd.DataFrame()

def load_infra_data(dataset_root: str, filename: str):
    """加载指标数据，支持从 dataset_root 或其父目录查找"""
    
    # 定义查找路径优先级
    paths_to_try = [
        os.path.join(dataset_root, 'processed', filename), # 优先找 processed
        os.path.join(dataset_root, filename),              # 其次找 root
        os.path.join(os.path.dirname(dataset_root.rstrip('/')), filename), # 找父目录 dataset/tianchi
    ]

    infra_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            infra_path = p
            break

    if not infra_path:
        print(f"⚠️ 警告: 未找到指标数据文件: {filename}")
        print(f"   请确保你已经运行了合并脚本，并将文件放在 {dataset_root} 或其父目录下")
        return None
    
    print(f"✅ 已加载指标数据: {infra_path}")
    try:
        df = pd.read_csv(infra_path)
        
        # 1. 检查关键列
        if 'timeMs' not in df.columns:
            if 'timestamp' in df.columns:
                df['timeMs'] = df['timestamp'].astype(np.int64) // 1000000
            else:
                print("❌ 错误: CSV缺少 'timeMs' 或 'timestamp' 列")
                return None
                
        if 'kubernetes_node' not in df.columns:
            if 'instance_id' in df.columns:
                df['kubernetes_node'] = df['instance_id'].astype(str)
            else:
                print("❌ 错误: CSV缺少 'kubernetes_node' 或 'instance_id' 列")
                return None

        # 2. 过滤需要的指标列
        # 兼容逻辑：如果 CSV 里已经是标准名(node_cpu...)就用标准名，否则用天池名
        valid_cols = []
        for m in TIANCHI_METRICS:
            if m in df.columns:
                valid_cols.append(m)
                df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0.0)
            # 这里可以加个 else 检查标准名，视你合并脚本的逻辑而定
        
        if not valid_cols:
            print("⚠️ 警告: 未找到任何匹配的指标列，请检查 CSV 表头")
            return None

        # 3. 构建索引 (按节点分组)
        cols = ['timeMs', 'kubernetes_node'] + valid_cols
        df = df[cols].dropna(subset=['timeMs', 'kubernetes_node'])
        
        host_idx = {}
        for host, g in tqdm(df.groupby('kubernetes_node'), desc="构建内存索引"):
            lg = g.sort_values('timeMs')
            lg = lg.drop_duplicates(subset=['timeMs'], keep='last')
            
            host_idx[str(host)] = {
                'timeMs': lg['timeMs'].to_numpy(dtype=np.int64),
                'metrics': {m: lg[m].to_numpy(dtype=np.float64) for m in valid_cols}
            }
        return host_idx
    except Exception as e:
        print(f"解析指标数据失败: {e}")
        return None

# ... (precompute_host_states 和 precompute_host_sequences 函数逻辑无需修改，保持原样即可) ...
# 为了完整性，这里简写保留结构，实际运行时请确保这两个函数在代码中
def precompute_host_states(trace_graphs, infra_index, id_manager, W=3):
    if infra_index is None: return
    metrics = TIANCHI_METRICS # 使用上面定义的列表
    per_metric_dims = 4
    feature_dim = len(metrics) * per_metric_dims
    zero_vec = np.zeros(feature_dim, dtype=np.float32)

    for graph in tqdm(trace_graphs, desc="预计算 HostState (GNN)"):
        try:
            st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
            if isinstance(st, (int, float)):
                v = float(st)
                t0_ms = int(v if v > 1e12 else v * 1000.0)
            else:
                t0_ms = 0
            t0_min_ms = (t0_ms // 60000) * 60000
            
            nodes_in_graph = [node for _, node in graph.iter_bfs() if node.host_id and node.host_id > 0]
            host_ids = set(node.host_id for node in nodes_in_graph)
            host_state_map = {}
            
            for hid in host_ids:
                hname = id_manager.host_id.rev(int(hid))
                if not hname or str(hname).lower() == 'nan':
                    host_state_map[hid] = zero_vec.copy()
                    continue

                vec = host_state_vector(hname, infra_index, t0_min_ms, metrics=metrics, W=W, per_metric_dims=per_metric_dims)
                if vec is not None:
                    host_state_map[hid] = vec
            
            if host_state_map:
                graph.data['precomputed_host_state'] = host_state_map
        except Exception:
            continue

def precompute_host_sequences(trace_graphs, infra_index, id_manager):
    if infra_index is None: return
    # 简单的列名映射，如果 CSV 列名已经是 aggregate_...，这里映射需要注意
    # 如果你的 CSV 列名是 aggregate_...，下面这个映射要确保能找到
    def _map_metric(alias: str) -> str:
        alias = str(alias).lower().strip()
        mapping = {
            'cpu': 'aggregate_node_cpu_usage',
            'mem': 'aggregate_node_memory_usage',
            'disk': 'aggregate_node_disk_io_usage',
            'net': 'aggregate_node_net_receive_packages_errors_per_minute',
            'tcp': 'aggregate_node_tcp_inuse_total_num'
        }
        # 如果 alias 在 mapping 里，返回对应的 aggregate 名；否则返回 alias 本身（防止 alias 已经是真实名）
        return mapping.get(alias, alias)
    
    metrics_cols = [_map_metric(a) for a in SEQ_METRICS]
    W = SEQ_WINDOW
    
    # ... (Robust norm logic) ...
    def _robust_norm(x):
        med = np.nanmedian(x)
        iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
        denom = iqr if iqr > 1e-6 else (np.nanstd(x) if np.nanstd(x) > 1e-6 else 1.0)
        return np.nan_to_num((x - med) / denom, nan=0.0)

    for graph in tqdm(trace_graphs, desc="预计算 HostSeq (OmniAnomaly)"):
        try:
            st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
            if isinstance(st, (int, float)):
                v = float(st)
                t0_ms = int(v if v > 1e12 else v * 1000.0)
            else:
                t0_ms = 0
            t0_min = (t0_ms // 60000) * 60000

            host_ids = set(node.host_id for _, node in graph.iter_bfs() if node.host_id and node.host_id > 0)
            host_seq_map = {}

            for hid in host_ids:
                hname = id_manager.host_id.rev(int(hid))
                if not hname: continue
                rec = infra_index.get(str(hname))
                if not rec: continue
                t_arr = rec.get('timeMs', [])
                if len(t_arr) == 0: continue
                
                per_metric = []
                for mcol in metrics_cols:
                    vals = rec.get('metrics', {}).get(mcol, [])
                    if len(vals) == 0:
                        seq_vals_np = np.zeros(W, dtype=np.float64)
                    else:
                        seq_vals = []
                        for k in range(W):
                            target = t0_min - (W - 1 - k) * 60000
                            pos = int(np.searchsorted(t_arr, target, side='right')) - 1
                            seq_vals.append(float(vals[pos]) if pos >= 0 else np.nan)
                        seq_vals_np = np.array(seq_vals, dtype=np.float64)
                    per_metric.append(_robust_norm(seq_vals_np).astype(np.float32))
                
                if per_metric:
                    host_seq_map[int(hid)] = torch.from_numpy(np.stack(per_metric, axis=1))
            
            if host_seq_map:
                graph.data['precomputed_host_seq'] = host_seq_map
        except Exception:
            continue

def process_split(split_name, dataset_root, id_manager, infra_index, processed_df=None):
    raw_csv = os.path.join(dataset_root, 'raw', f'{split_name}.csv')
    out_dir = os.path.join(dataset_root, 'processed', split_name)
    if not os.path.exists(raw_csv) and processed_df is None: return

    print(f"\n=== 处理 {split_name} 集 ===")
    os.makedirs(out_dir, exist_ok=True)
    
    if processed_df is not None: df = processed_df
    else: df = flexible_load_trace_csv(raw_csv)

    if df.empty: return

    trace_graphs = df_to_trace_graphs(df=df, id_manager=id_manager, min_node_count=2, max_node_count=100, summary_file=None, merge_spans=False)
    if not trace_graphs: return

    # === 执行两项预计算 ===
    # 传入同一个 infra_index
    precompute_host_states(trace_graphs, infra_index, id_manager)    
    precompute_host_sequences(trace_graphs, infra_index, id_manager) 
    # ====================

    db_path = os.path.join(out_dir, "_bytes.db")
    if not os.path.exists(db_path): open(db_path, 'a').close()
    db = TraceGraphDB(BytesSqliteDB(out_dir, write=True))
    try:
        with db.write_batch():
            for graph in trace_graphs:
                if hasattr(graph, 'root_cause') and graph.root_cause is None: graph.root_cause = 0
                if hasattr(graph, 'fault_category') and graph.fault_category is None: graph.fault_category = 0
                db.add(graph)
        db.commit()
        print(f"  ✅ 成功写入 {len(trace_graphs)} 个图到 {split_name} 数据库")
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args()
    
    dataset_root = args.root
    print(f"🚀 开始处理数据流 (单文件模式)，根目录: {dataset_root}")
    processed_root = os.path.join(dataset_root, 'processed')
    os.makedirs(processed_root, exist_ok=True)
    
    # 1. 加载唯一的指标文件
    print(f"\n[步骤 0/4] 加载指标文件 {INFRA_FILENAME}...")
    global_infra_index = load_infra_data(dataset_root, INFRA_FILENAME)
    
    # 2. 建立 ID 映射 (保持不变)
    print("\n[步骤 1/4] 建立统一 ID 映射...")
    combined_dfs = []
    for split in ['train', 'val', 'test']:
        path = os.path.join(dataset_root, 'raw', f'{split}.csv')
        df = flexible_load_trace_csv(path)
        if not df.empty: combined_dfs.append(df)
    if not combined_dfs: return
    temp_id_dir = os.path.join(dataset_root, 'temp_ids')
    os.makedirs(temp_id_dir, exist_ok=True)
    id_manager = TraceGraphIDManager(temp_id_dir)
    with id_manager:
        full_df = pd.concat(combined_dfs, ignore_index=True)
        for row in tqdm(full_df.itertuples(), total=len(full_df), desc="生成 ID"):
            id_manager.service_id.get_or_assign(getattr(row, 'ServiceName', '') or '')
            id_manager.operation_id.get_or_assign(getattr(row, 'OperationName', '') or '')
            id_manager.status_id.get_or_assign(str(getattr(row, 'StatusCode', '')) or '')
    id_manager.dump_to(processed_root)
    id_manager = TraceGraphIDManager(processed_root)
    if os.path.exists(temp_id_dir): shutil.rmtree(temp_id_dir)

    # 3. 处理数据 (Train/Val/Test 统一使用 global_infra_index)
    process_split('train', dataset_root, id_manager, global_infra_index)
    process_split('val', dataset_root, id_manager, global_infra_index)

    print("\n[步骤 3/4] 处理测试集...")
    test_csv_path = os.path.join(dataset_root, 'raw', 'test.csv')
    test_df = flexible_load_trace_csv(test_csv_path)
    if not test_df.empty:
        # ID 映射逻辑...
        for col in ['RootCause', 'FaultCategory']:
            if col not in test_df.columns: test_df[col] = ''
        for idx, row in test_df.iterrows():
            if row.get('Anomaly'):
                rc_text = str(row.get('RootCause', '')).strip()
                fc_text = str(row.get('FaultCategory', '')).strip()
                mapped_id = None
                if fc_text.lower().startswith('node'):
                    rc_text = rc_text.replace('_', '-')
                    mapped_id = id_manager.host_id.get(rc_text)
                else:
                    rc_svc = rc_text.split('-')[0] if '-' in rc_text else rc_text
                    mapped_id = id_manager.service_id.get(rc_svc)
                test_df.at[idx, 'RootCause'] = mapped_id if mapped_id is not None else 0
                test_df.at[idx, 'FaultCategory'] = id_manager.fault_category.get_or_assign(fc_text) if fc_text else 0
        
        process_split('test', dataset_root, id_manager, global_infra_index, processed_df=test_df)

    # 4. [关键] 保存索引文件到磁盘！
    if global_infra_index:
        pkl_path = os.path.join(processed_root, 'host_infra_index.pkl')
        print(f"\n[步骤 4/4] 💾 保存指标索引到 PKL: {pkl_path}")
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(global_infra_index, f)
            print("  ✅ 索引保存成功 (评估脚本可以直接读取了)")
        except Exception as e:
            print(f"  ❌ 索引保存失败: {e}")

    id_manager.dump_to(processed_root)
    print(f"\n✨ 所有处理完成！")

if __name__ == '__main__':
    main()