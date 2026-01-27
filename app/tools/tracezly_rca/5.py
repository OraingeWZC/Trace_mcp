import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import re
import os
import sqlite3
from tracegnn.data.trace_graph import df_to_trace_graphs, TraceGraphIDManager
from tracegnn.data.trace_graph_db import TraceGraphDB, BytesSqliteDB

# 根据划分好的数据集，进行数据处理


def flexible_load_trace_csv(csv_file: str) -> pd.DataFrame:
    """加载CSV文件"""
    print(f"正在加载CSV文件: {csv_file}")
    df = pd.read_csv(
        csv_file,
        engine='python',
        encoding='utf-8',
        on_bad_lines='warn'
    )
    unique_traces = df['TraceID'].nunique() if 'TraceID' in df.columns else 0
    print(f"成功加载 {len(df)} 行数据，包含 {unique_traces} 个唯一的TraceID")
    return df


def preprocess_fault_and_service_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理FaultCategory字段：
    - 所有无效值（None/NaN/''/'0'/0） → 转为空字符串 ''
    - ServiceName: 保持原逻辑，转为数字 0
    """
    df = df.copy()

    # 处理 FaultCategory → 转为 ''
    if 'FaultCategory' in df.columns:
        print("正在处理FaultCategory字段...")
        original_counts = df['FaultCategory'].value_counts(dropna=False)
        print(f"原始分布:")
        for val, count in original_counts.head(10).items():
            print(f"  {repr(val)}: {count}")

        def clean_fault(x):
            if pd.isna(x):
                return ''
            x_str = str(x).strip()
            if x_str == '' or x_str == '0' or x_str.lower() == 'nan':
                return ''
            return x_str

        df['FaultCategory'] = df['FaultCategory'].apply(clean_fault)

        processed_counts = df['FaultCategory'].value_counts(dropna=False)
        print(f"处理后分布:")
        for val, count in processed_counts.head(10).items():
            print(f"  {repr(val)}: {count}")
        print(f"映射为 '' 的数量: {(df['FaultCategory'] == '').sum()}")

    # 处理 ServiceName → 转为 0
    if 'ServiceName' in df.columns:
        print("正在处理ServiceName字段...")

        def clean_service(x):
            if pd.isna(x):
                return 0
            x_str = str(x).strip()
            if x_str == '' or x_str == '0':
                return 0
            return x

        df['ServiceName'] = df['ServiceName'].apply(clean_service)

    return df


def preprocess_root_cause(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理RootCause字段：
    1. 去掉 -数字 后缀
    2. 所有无效值 → 转为空字符串 ''
    """
    df = df.copy()

    if 'RootCause' in df.columns:
        print("正在处理RootCause字段...")
        original_counts = df['RootCause'].value_counts(dropna=False)
        print(f"原始分布:")
        for val, count in original_counts.head(10).items():
            print(f"  {repr(val)}: {count}")

        def clean_root_cause(x):
            if pd.isna(x):
                return ''
            x_str = str(x).strip()
            if x_str == '' or x_str == '0' or x_str.lower() == 'nan':
                return ''

            # 只对有效值去除后缀
            x_clean = re.sub(r'-\d+$', '', x_str)
            return x_clean

        df['RootCause'] = df['RootCause'].apply(clean_root_cause)

        processed_counts = df['RootCause'].value_counts(dropna=False)
        print(f"处理后分布:")
        for val, count in processed_counts.head(10).items():
            print(f"  {repr(val)}: {count}")
        print(f"映射为 '' 的数量: {(df['RootCause'] == '').sum()}")

    return df


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    自动填充缺失值：
    - 数值列 → 0
    - 其他（字符串/对象等）→ 空字符串 ''
    返回填充后的新 DataFrame。
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('')
    return df


def assert_no_nan(df: pd.DataFrame, name: str):
    """
    检查整个 DataFrame 是否存在 NaN/None，发现后直接报错，防止脏数据进入后续图转换与训练。
    """
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        details = ", ".join([f"{col}: {cnt}" for col, cnt in nan_cols.items()])
        raise ValueError(f"[NaN Check] {name} 数据集中存在 NaN 值: {details}")


def get_trace_statistics(df: pd.DataFrame) -> dict:
    """获取trace统计信息"""
    stats = {}
    if 'TraceID' in df.columns:
        stats['total_rows'] = len(df)
        stats['unique_traces'] = df['TraceID'].nunique()
        stats['avg_spans_per_trace'] = len(df) / df['TraceID'].nunique() if df['TraceID'].nunique() > 0 else 0
        spans_per_trace = df.groupby('TraceID').size()
        stats['min_spans_per_trace'] = spans_per_trace.min()
        stats['max_spans_per_trace'] = spans_per_trace.max()
        stats['median_spans_per_trace'] = spans_per_trace.median()
    else:
        stats.update({'total_rows': 0, 'unique_traces': 0, 'avg_spans_per_trace': 0,
                      'min_spans_per_trace': 0, 'max_spans_per_trace': 0, 'median_spans_per_trace': 0})
    return stats


def process_dataset_to_graphs_and_db(
    df: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
    trace_graph_id_manager: TraceGraphIDManager,
    min_node_count: int = 1,
    max_node_count: int = 1000
):
    """处理单个数据集"""
    print(f"\n正在处理{dataset_name}数据集...")
    os.makedirs(output_dir, exist_ok=True)

    input_stats = get_trace_statistics(df)
    print(f"输入: {input_stats['total_rows']} 行, {input_stats['unique_traces']} 个TraceID")

    # 关键：确保字段中没有 None，全部是 '' 或有效字符串
    if 'RootCause' in df.columns:
        assert not df['RootCause'].isnull().any(), "RootCause 中存在 None！"
    if 'FaultCategory' in df.columns:
        assert not df['FaultCategory'].isnull().any(), "FaultCategory 中存在 None！"

    print("正在将数据转换为图结构...")
    trace_graphs = df_to_trace_graphs(
        df=df,
        id_manager=trace_graph_id_manager,
        min_node_count=min_node_count,
        max_node_count=max_node_count,
        summary_file=os.path.join(output_dir, f'{dataset_name}_conversion_summary.txt'),
        merge_spans=False
    )

    trace_graphs = trace_graphs or []

    print(f"成功转换 {len(trace_graphs)} 个图")

    if len(trace_graphs) > 0:
        db_path = os.path.join(output_dir, "_bytes.db")
        if not os.path.exists(db_path):
            print(f"创建数据库: {db_path}")
            conn = sqlite3.connect(db_path)
            conn.close()

        db = TraceGraphDB(BytesSqliteDB(output_dir, write=True))
        print(f"写入数据库...")
        try:
            with db.write_batch():
                for graph in trace_graphs:
                    db.add(graph)
            db.commit()
            print(f"成功写入 {len(trace_graphs)} 个图")
        finally:
            db.close()

    return trace_graphs, input_stats


def convert_csv_to_db(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    output_root: str,
    min_node_count: int = 1,
    max_node_count: int = 1000
):
    """完整处理流程"""
    print("=== 开始数据处理流水线 ===")

    print("步骤1: 加载数据集...")
    train_df = flexible_load_trace_csv(train_csv)
    val_df = flexible_load_trace_csv(val_csv)
    test_df = flexible_load_trace_csv(test_csv)

    print("\n步骤2: 创建统一ID管理器（合并 train/val/test 共用映射，先清空旧映射文件）...")
    # 清空旧的 ID 映射，避免历史残留导致 train/val/test 不一致
    for fname in ["service_id.yml", "operation_id.yml", "status_id.yml", "fault_category.yml"]:
        fpath = os.path.join(output_root, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    trace_graph_id_manager = TraceGraphIDManager(output_root)

    print("\n步骤3: 预处理数据...")
    train_df = preprocess_fault_and_service_fields(train_df)
    train_df = preprocess_root_cause(train_df)
    train_df = fill_na(train_df)  # 自动填充 NaN
    assert_no_nan(train_df, "train")  # 确认填充后无 NaN

    val_df = preprocess_fault_and_service_fields(val_df)
    val_df = preprocess_root_cause(val_df)
    val_df = fill_na(val_df)
    assert_no_nan(val_df, "val")

    test_df = preprocess_fault_and_service_fields(test_df)
    test_df = preprocess_root_cause(test_df)
    test_df = fill_na(test_df)
    assert_no_nan(test_df, "test")

    # ✅ 强制检查是否还有 None
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if 'RootCause' in df.columns:
            null_count = df['RootCause'].isnull().sum()
            assert null_count == 0, f"{name} 中 RootCause 存在 {null_count} 个 None！请检查预处理逻辑。"
        if 'FaultCategory' in df.columns:
            null_count = df['FaultCategory'].isnull().sum()
            assert null_count == 0, f"{name} 中 FaultCategory 存在 {null_count} 个 None！"

    print("\n步骤4: 转换为图结构...")
    datasets = [
        ("train", train_df, os.path.join(output_root, "train")),
        ("val", val_df, os.path.join(output_root, "val")),
        ("test", test_df, os.path.join(output_root, "test"))
    ]

    all_graphs = {}
    all_input_stats = {}
    for name, df_subset, output_dir in datasets:
        graphs, input_stats = process_dataset_to_graphs_and_db(
            df=df_subset,
            output_dir=output_dir,
            dataset_name=name,
            trace_graph_id_manager=trace_graph_id_manager,
            min_node_count=min_node_count,
            max_node_count=max_node_count
        )
        all_graphs[name] = graphs
        all_input_stats[name] = input_stats

    print(f"\n步骤5: 保存ID映射文件...")
    trace_graph_id_manager.dump_to(output_root)
    print(f"已完成ID映射保存: {output_root}")

    print(f"\n=== 处理完成 ===")
    total_input = sum(s['unique_traces'] for s in all_input_stats.values())
    total_output = sum(len(g) for g in all_graphs.values())
    print(f"总计: {total_input} TraceID → {total_output} 图")

    return all_graphs, all_input_stats


def main():
    train_csv_path = "/mnt/sdb/zly/4.1/tracezly_rca/processed/tianchi_processed_data2/train.csv"
    val_csv_path = "/mnt/sdb/zly/4.1/tracezly_rca/processed/tianchi_processed_data2/val.csv"
    test_csv_path = "/mnt/sdb/zly/4.1/tracezly_rca/processed/tianchi_processed_data2/test.csv"
    output_root = "/mnt/sdb/zly/4.1/tracezly_rca/dataset/tianchi2/processed"

    graphs, stats = convert_csv_to_db(
        train_csv_path,
        val_csv_path,
        test_csv_path,
        output_root,
        min_node_count=2,
        max_node_count=100
    )
    return graphs, stats


if __name__ == "__main__":
    main()