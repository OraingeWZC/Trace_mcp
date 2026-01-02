import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import shutil
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm

# 引入项目依赖
from tracegnn.data.trace_graph import df_to_trace_graphs, TraceGraphIDManager
from tracegnn.data.trace_graph_db import TraceGraphDB, BytesSqliteDB
from tracegnn.utils.host_state import host_state_vector, DEFAULT_METRICS, DISK_METRICS

# ================= 配置区域 =================
# 默认的数据集根目录，您可以在这里修改，或者通过命令行参数 --root 传入
DEFAULT_DATASET_ROOT = 'dataset/dataset_topo' 
INFRA_FILENAME = 'merged_all_infra.csv'
# ===========================================

def flexible_load_trace_csv(input_path: str) -> pd.DataFrame:
    """更灵活地加载CSV文件"""
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(input_path)
        # 类型转换兜底
        if 'Duration' in df.columns:
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
        if 'StartTimeMs' in df.columns:
            df['StartTime'] = pd.to_numeric(df['StartTimeMs'], errors='coerce')
        if 'Anomaly' in df.columns:
            df['Anomaly'] = df['Anomaly'].astype(bool)
        return df
    except Exception as e:
        print(f"加载CSV出错 {input_path}: {e}")
        return pd.DataFrame()

def load_infra_data_from_parent(dataset_root: str):
    """
    从数据集根目录的上一级查找指标数据
    例如: root = 'dataset/dataset_demo' -> 查找 'dataset/merged_all_infra.csv'
    """
    parent_dir = os.path.dirname(dataset_root.rstrip(os.path.sep))
    infra_path = os.path.join(parent_dir, INFRA_FILENAME)
    
    # 如果上一级找不到，尝试上一级的 infra 目录
    if not os.path.exists(infra_path):
        infra_path_alt = os.path.join(parent_dir, 'infra', INFRA_FILENAME)
        if os.path.exists(infra_path_alt):
            infra_path = infra_path_alt
    
    if not os.path.exists(infra_path):
        print(f"⚠️ 警告: 未找到指标数据文件。期望路径: {infra_path}")
        return None
    
    print(f"✅ 已加载指标数据: {infra_path}")
    
    # 加载并构建索引
    try:
        df = pd.read_csv(infra_path)
        if 'timeMs' not in df.columns or 'kubernetes_node' not in df.columns:
            return None
            
        # 确保包含需要的指标列
        all_metrics = list(set(DEFAULT_METRICS + DISK_METRICS))
        for m in all_metrics:
            if m not in df.columns:
                df[m] = np.nan
        
        try:
            df['timeMs'] = df['timeMs'].astype(np.int64)
        except:
            if 'time' in df.columns:
                df['timeMs'] = pd.to_datetime(df['time']).astype('int64') // 10**6
        
        cols = ['timeMs', 'kubernetes_node'] + [c for c in all_metrics if c in df.columns]
        df = df[cols].dropna(subset=['timeMs', 'kubernetes_node'])
        
        host_idx = {}
        for host, g in df.groupby('kubernetes_node'):
            lg = g.sort_values('timeMs')
            host_idx[str(host)] = {
                'timeMs': lg['timeMs'].to_numpy(dtype=np.int64),
                'metrics': {m: lg[m].to_numpy(dtype=np.float64) for m in lg.columns if m not in ('timeMs', 'kubernetes_node')}
            }
        return host_idx
    except Exception as e:
        print(f"解析指标数据失败: {e}")
        return None

def precompute_host_states(trace_graphs, infra_index, id_manager, W=3):
    """预计算 SInfra 数据并注入到图对象中"""
    if infra_index is None:
        return

    metrics = list(DEFAULT_METRICS)
    # 如果需要磁盘指标，取消注释
    # for m in DISK_METRICS:
    #     if m not in metrics: metrics.append(m)
    per_metric_dims = 3

    success_cnt = 0
    for graph in tqdm(trace_graphs, desc="预计算 SInfra (HostState)"):
        try:
            # 1. 计算 t0
            st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
            if isinstance(st, (int, float)):
                v = float(st)
                t0_ms = int(v if v > 1e12 else v * 1000.0)
            else:
                t0_ms = 0
            t0_min_ms = (t0_ms // 60000) * 60000

            # 2. 查找涉及的主机
            host_ids = set(node.host_id for _, node in graph.iter_bfs() if node.host_id and node.host_id > 0)
            
            # 3. 计算向量
            host_state_map = {}
            for hid in host_ids:
                hname = id_manager.host_id.rev(int(hid))
                if hname:
                    vec = host_state_vector(hname, infra_index, t0_min_ms, metrics=metrics, W=W, per_metric_dims=per_metric_dims)
                    if vec is not None:
                        host_state_map[hid] = vec
            
            if host_state_map:
                graph.data['precomputed_host_state'] = host_state_map
                success_cnt += 1
        except Exception:
            continue
    
    print(f"  -> SInfra 预计算完成: {success_cnt}/{len(trace_graphs)} 个 Trace 包含主机指标数据")

def process_split(split_name, dataset_root, id_manager, infra_index, processed_df=None):
    """处理单个数据集分片 (train/val/test)"""
    raw_csv = os.path.join(dataset_root, 'raw', f'{split_name}.csv')
    out_dir = os.path.join(dataset_root, 'processed', split_name)
    
    if not os.path.exists(raw_csv):
        print(f"跳过 {split_name}: 文件不存在 {raw_csv}")
        return

    print(f"\n=== 处理 {split_name} 集 ===")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 加载数据
    if processed_df is not None:
        df = processed_df
        print(f"  使用预处理后的 DataFrame ({len(df)} 行)")
    else:
        df = flexible_load_trace_csv(raw_csv)
        print(f"  加载 CSV: {len(df)} 行")

    if df.empty:
        return

    # 2. 转换为图
    trace_graphs = df_to_trace_graphs(
        df=df,
        id_manager=id_manager,
        min_node_count=2,
        max_node_count=100,
        summary_file=None,
        merge_spans=False
    )
    
    if not trace_graphs:
        print("  没有生成有效的 Trace Graph")
        return

    # 3. 预计算 SInfra
    precompute_host_states(trace_graphs, infra_index, id_manager)

    # 4. 写入数据库
    db_path = os.path.join(out_dir, "_bytes.db")
    # 确保文件存在
    if not os.path.exists(db_path):
        open(db_path, 'a').close()
        
    db = TraceGraphDB(BytesSqliteDB(out_dir, write=True))
    try:
        with db.write_batch():
            for graph in trace_graphs:
                # 确保类型安全
                if hasattr(graph, 'root_cause') and graph.root_cause is None: graph.root_cause = 0
                if hasattr(graph, 'fault_category') and graph.fault_category is None: graph.fault_category = 0
                db.add(graph)
        db.commit()
        print(f"  ✅ 成功写入 {len(trace_graphs)} 个图到 {split_name} 数据库")
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description="Trace数据处理流水线 (v2)")
    parser.add_argument('--root', type=str, default=DEFAULT_DATASET_ROOT, 
                        help='数据集根目录 (例如: dataset/dataset_demo)')
    args = parser.parse_args()
    
    dataset_root = args.root
    print(f"🚀 开始处理数据流，根目录: {dataset_root}")
    
    # 0. 准备目录
    processed_root = os.path.join(dataset_root, 'processed')
    os.makedirs(processed_root, exist_ok=True)
    
    # 1. 加载指标数据 (优化点: 只加载一次)
    infra_index = load_infra_data_from_parent(dataset_root)
    
    # 2. 建立统一的 ID 映射 (Train + Val + Test)
    print("\n[步骤 1/4] 建立统一 ID 映射...")
    combined_dfs = []
    for split in ['train', 'val', 'test']:
        path = os.path.join(dataset_root, 'raw', f'{split}.csv')
        df = flexible_load_trace_csv(path)
        if not df.empty:
            combined_dfs.append(df)
            
    if not combined_dfs:
        print("❌ 没有找到任何 CSV 数据，退出。")
        return

    # 使用临时目录生成 ID，然后移动到 processed
    temp_id_dir = os.path.join(dataset_root, 'temp_ids')
    os.makedirs(temp_id_dir, exist_ok=True)
    id_manager = TraceGraphIDManager(temp_id_dir)
    
    with id_manager:
        full_df = pd.concat(combined_dfs, ignore_index=True)
        for row in tqdm(full_df.itertuples(), total=len(full_df), desc="生成 ID"):
            id_manager.service_id.get_or_assign(getattr(row, 'ServiceName', '') or '')
            id_manager.operation_id.get_or_assign(getattr(row, 'OperationName', '') or '')
            id_manager.status_id.get_or_assign(str(getattr(row, 'StatusCode', '')) or '')
            # 注意: FaultCategory 和 HostID 也会在 df_to_trace_graphs 中动态添加

    # 将 ID 文件保存到最终目录
    id_manager.dump_to(processed_root)
    # 重新初始化指向最终目录的 manager
    id_manager = TraceGraphIDManager(processed_root)
    
    # 清理临时目录
    if os.path.exists(temp_id_dir):
        shutil.rmtree(temp_id_dir)

    # 3. 处理 Train 和 Val
    print("\n[步骤 2/4] 处理训练集和验证集...")
    process_split('train', dataset_root, id_manager, infra_index)
    process_split('val', dataset_root, id_manager, infra_index)

    # 4. 特殊处理测试集 (映射 RootCause 和 FaultCategory)
    print("\n[步骤 3/4] 处理测试集 (包含故障映射)...")
    test_csv_path = os.path.join(dataset_root, 'raw', 'test.csv')
    test_df = flexible_load_trace_csv(test_csv_path)
    
    if not test_df.empty:
        # 处理故障映射逻辑
        print("  正在执行测试集故障文本映射...")
        # 确保列存在
        for col in ['RootCause', 'FaultCategory']:
            if col not in test_df.columns: test_df[col] = ''
            
        for idx, row in test_df.iterrows():
            if row.get('Anomaly'):
                rc_text = str(row.get('RootCause', '')).strip()
                fc_text = str(row.get('FaultCategory', '')).strip()
                
                # 映射 RootCause -> ID
                mapped_id = None
                if fc_text.lower().startswith('node'):
                    rc_text = rc_text.replace('_', '-')
                    mapped_id = id_manager.host_id.get(rc_text)
                else:
                    rc_svc = rc_text.split('-')[0] if '-' in rc_text else rc_text
                    mapped_id = id_manager.service_id.get(rc_svc)
                
                test_df.at[idx, 'RootCause'] = mapped_id if mapped_id is not None else 0
                
                # 映射 FaultCategory -> ID
                if fc_text:
                    fc_id = id_manager.fault_category.get_or_assign(fc_text)
                    test_df.at[idx, 'FaultCategory'] = fc_id
                else:
                    test_df.at[idx, 'FaultCategory'] = 0

        # 处理并写入测试集
        process_split('test', dataset_root, id_manager, infra_index, processed_df=test_df)

    # 5. 收尾
    print("\n[步骤 4/4] 保存最终映射文件...")
    id_manager.dump_to(processed_root)
    
    print(f"\n✨ 所有处理完成！输出目录: {processed_root}")

if __name__ == '__main__':
    main()