import os
import argparse
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def load_mapping(mapping_path):
    if not mapping_path or not os.path.exists(mapping_path):
        print(f"⚠️  未找到映射文件: {mapping_path} (将跳过 IP->ID 转换尝试)")
        return {}
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 合并 ip_to_id 和 name_to_id 为一个大字典
    lookup = {}
    if "ip_to_id" in data: lookup.update(data["ip_to_id"])
    if "name_to_id" in data: lookup.update(data["name_to_id"])
    
    print(f"✅ 已加载映射表，包含 {len(lookup)} 个映射规则")
    return lookup

def check_data_quality(trace_file, metric_file, mapping_file=None):
    print(f"🚀 开始诊断数据质量...")
    print(f"   Trace 文件: {trace_file}")
    print(f"   Metric 文件: {metric_file}")
    
    # 1. 加载 Trace 节点信息
    if not os.path.exists(trace_file):
        print("❌ Trace 文件不存在")
        return
    
    print("⏳ 正在读取 Trace 文件 (可能较慢)...")
    # 只读需要的列，加快速度
    df_trace = pd.read_csv(trace_file, usecols=['TraceID', 'NodeName', 'StartTimeMs', 'EndTimeMs'])
    
    # 统计 Trace 涉及的唯一节点
    trace_nodes = df_trace['NodeName'].unique()
    trace_nodes = [str(n).strip() for n in trace_nodes if pd.notnull(n) and str(n).strip() != '']
    print(f"   -> Trace 中共发现 {len(trace_nodes)} 个独立节点标识 (NodeName)")

    # 2. 加载 Metric 节点信息
    if not os.path.exists(metric_file):
        print("❌ Metric 文件不存在")
        return
    
    print("⏳ 正在读取 Metric 文件...")
    df_metric = pd.read_csv(metric_file, usecols=['instance_id', 'timestamp'])
    if 'instanceId' in df_metric.columns: 
        df_metric.rename(columns={'instanceId': 'instance_id'}, inplace=True)
        
    metric_nodes = set(df_metric['instance_id'].astype(str).unique())
    print(f"   -> Metric 中共包含 {len(metric_nodes)} 个物理机 (Instance ID)")
    
    # 3. 加载映射表
    mapping = load_mapping(mapping_file)
    
    # === 开始诊断 ===
    print("\n🔍 === 诊断报告 ===")
    
    results = {
        "success": [],       # 成功：Trace节点 -> 映射ID -> Metric中有数据
        "no_mapping": [],    # 失败：Trace节点是IP，且映射表中找不到ID
        "no_metric": [],     # 失败：Trace节点(或映射后)是ID，但Metric表中没数据
        "time_mismatch": []  # 警告：有ID也有Metric，但Trace发生时Metric没覆盖 (暂未详细实现，仅作提示)
    }
    
    for original_name in tqdm(trace_nodes, desc="检查节点"):
        final_id = original_name
        is_mapped = False
        
        # 步骤 A: 尝试映射
        # 如果原始名字就像一个 ID (i-开头)，则直接使用
        if original_name.startswith('i-'):
            final_id = original_name
        else:
            # 尝试从映射表中查找
            if original_name in mapping:
                final_id = mapping[original_name]
                is_mapped = True
            else:
                # 映射失败
                results["no_mapping"].append(original_name)
                continue
        
        # 步骤 B: 检查 Metric 是否存在
        if final_id in metric_nodes:
            results["success"].append(f"{original_name} -> {final_id}")
        else:
            results["no_metric"].append(f"{original_name} -> {final_id}")

    # === 输出汇总 ===
    total = len(trace_nodes)
    print("\n📊 统计结果:")
    print(f"   ✅ 完全匹配成功: {len(results['success'])} / {total} ({(len(results['success'])/total)*100:.1f}%)")
    print(f"   ❌ 映射失败 (缺字典): {len(results['no_mapping'])}")
    print(f"   ❌ 指标缺失 (有ID无数据): {len(results['no_metric'])}")
    
    if results["no_mapping"]:
        print(f"\n⚠️  [Top 5] 映射失败的节点 (请检查 data/ecs_mapping_index.json):")
        for x in results["no_mapping"][:5]: print(f"   - {x}")
        
    if results["no_metric"]:
        print(f"\n⚠️  [Top 5] 有ID但无指标的节点 (请检查 Metric 文件是否覆盖了对应机器):")
        for x in results["no_metric"][:5]: print(f"   - {x}")

    # 建议
    print("\n💡 修复建议:")
    if len(results["no_mapping"]) > 0:
        print("   1. 你的 ecs_mapping_index.json 可能过期了，或者 Trace 里的 IP 是新的/临时的。")
    if len(results["no_metric"]) > 0:
        print("   2. 2_get_normalData.py 可能漏抓了这些节点。")
        print("      建议：不要使用 'Trace导向' (fetch_metrics带target_nodes) 的方式，")
        print("      而是改回 '全量抓取' (只指定时间窗)，让它把该时间段内所有活跃节点的数据都抓下来。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认值根据你之前的文件名设定
    parser.add_argument("--trace", default="data/NormalData/normal_traces1e5_30s.csv", help="Trace CSV 路径")
    parser.add_argument("--metric", default="data/NormalData/normal_metrics_1e5_30s.csv", help="Metric CSV 路径")
    parser.add_argument("--mapping", default="data/ecs_mapping_index.json", help="映射文件路径")
    
    args = parser.parse_args()
    
    check_data_quality(args.trace, args.metric, args.mapping)